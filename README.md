# ER Pulse

Live dashboard: (Hugging Face Space link, added Nov 22)

Audience: ED operations and patient flow managers in Metro Vancouver deciding when to add staff and when to start "consider an urgent care centre" messaging.

Wait times are for general information only. In an emergency, call 911.

## What runs where

| Job | Script | Schedule | Writes |
|---|---|---|---|
| Collector | `collector/run_loop.py` | GitHub Actions, every 15 min | ed_waits, facilities, raw_polls, poll_log, weather_hourly |
| Model job | `model/retrain_and_predict.py` | GitHub Actions, hourly | predictions, model_runs |
| Dashboard | `app.py` (to do) | Hugging Face Space | reads only |

All three share one Postgres database (Neon). Without `DATABASE_URL` the dashboard falls back to `data/sample.db`.

## Start collecting (one time setup, about 20 minutes)

1. **Database.** Create a free project at neon.tech (region: US West if offered). Copy the connection string from Connect. Keep `?sslmode=require` on the end.
   If you use Supabase instead, copy the **Session pooler** string, not the direct one (GitHub runners cannot reach the direct host).
2. **Repo.** Create a **public** GitHub repo (public keeps Actions minutes free) and push this folder to the `main` branch.
3. **Secrets.** Repo Settings, Secrets and variables, Actions, New repository secret:
   `DATABASE_URL` = the connection string, `CONTACT_EMAIL` = a team email.
4. **Create tables and load the static lists** from your own machine:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (Windows)   or   source .venv/bin/activate
   pip install -r requirements-collector.txt
   copy .env.example .env           (Windows)   or   cp .env.example .env
   (edit .env and paste the same DATABASE_URL)
   python scripts/init_db.py
   python collector/run_loop.py --inspect
   ```
   The last command does one real poll and prints the fields the source sends. You should see about 41 facilities and 10 to 15 with a wait time.
5. **Turn on the schedule.** Actions tab, enable workflows if asked, open **collect**, Run workflow. A green tick means it works. After that it runs on its own every 15 minutes.
6. **Check daily:** `python scripts/check_gaps.py`. GitHub also emails whoever last edited the workflow when a run fails.

## Triggering the jobs reliably

GitHub's own schedule did not fire on this repo, so cron-job.org triggers both workflows through the GitHub API (POST, headers `Accept: application/vnd.github+json`, `Authorization: Bearer <token>`, `X-GitHub-Api-Version: 2022-11-28`, body `{"ref":"main"}`):

| cron-job.org job | URL ends with | Schedule |
|---|---|---|
| er-pulse collect | `/actions/workflows/collect.yml/dispatches` | every 15 min |
| er-pulse model | `/actions/workflows/model.yml/dispatches` | every hour at :12 |

Full URL: `https://api.github.com/repos/OWNER/REPO/actions/workflows/<file>/dispatches`. A test run should return 204.

## Model job

Every hour it scores past predictions (printed in the run log), retrains, and writes a forecast 1, 2 and 3 hours ahead for every site with a fresh wait:

| model_name | What it is |
|---|---|
| persistence | Wait later = wait now |
| seasonal | Mean wait for that site at that hour of day, data so far |
| main | Gradient boosted trees predicting the change from the current wait, retrained hourly on all data, then bias corrected by the site's mean error over the last 24 h (`predicted_raw` keeps the uncorrected value) |
| day1_frozen | The same model trained only on the first 24 hours and never updated, to show how much retraining helps |

Predictions are append only and scored against the wait observed nearest the target time (within 15 min) by `model.features.score()`. Features: current wait, wait 1 h ago, change over the hour, 3 h mean, same time yesterday, hour now and at target, weekend or holiday, site, authority, ED or UPCC, temperature and rain at the target hour, weekly flu level. Weekday is left out until there is a full week of data.

Model code freezes on Nov 24 so the correction chart compares one model over time.

## Data sources and credits

| Source | What | How often | Access |
|---|---|---|---|
| edwaittimes.ca (VCH, Providence, Fraser Health, PHSA) | Posted wait per ED and UPCC | Every 15 min | Public JSON at `/api/wait-times` that the site itself uses |
| Open-Meteo forecast API | Hourly temperature, precipitation, wind for Vancouver | Every 15 min (upserted) | Free, no key, CC BY 4.0 |
| BCCDC respiratory virus data | Weekly flu, RSV, COVID activity level | Weekly, Thursdays | Typed into `data/static/resp_activity.csv` |
| Facility list | Authority and Metro Vancouver flag per site | Once | `data/static/facilities.csv` |

**Each Thursday:** add the new BCCDC week to `resp_activity.csv` (levels: 1 low, 1.5 low to moderate, 2 moderate, 2.5 moderate to high, 3 high, 4 very high), then run `python scripts/load_static.py`.

## Database schema

`facilities` (facility_id PK) joins to `ed_waits` (facility_id, observed_at). `weather_hourly` joins on the hour of observed_at. `resp_activity` joins on the week containing observed_at. `predictions` and `model_runs` are created now and filled by the model job. All times are UTC; convert to Pacific only for display.

## Testing offline

```
python -m pytest tests -q
python tests/simulate_data.py .                                   (2 days of simulated waits in data/local.db)
python model/retrain_and_predict.py --local --now 2026-11-21T12:00Z
python scripts/init_db.py --local
python collector/run_loop.py --local --from-file tests/fixture_wait_times.json
python scripts/check_gaps.py --local
```

## Known limitations

* Posted waits are the health authorities' own estimates, not measured waits.
* As of November 2026 only the VCH, Providence and BC Children's emergency departments post a wait. Fraser Health sites and the urgent care centres show "unavailable". The collector stores every site so we capture them if they come back.
* History starts the day the collector started (Nov 20, 2026).
* GitHub can delay scheduled runs; gaps are logged in `poll_log`, never filled with made up values.
