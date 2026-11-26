"""ER Pulse dashboard (Gradio). Read only: the collector and model job write the data.

  python app.py         opens http://localhost:7860
Uses DATABASE_URL if set, otherwise data/sample.db (offline mode).
"""
from __future__ import annotations

import os

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from db import get_engine, metadata  # noqa: E402

from dashboard import charts as C  # noqa: E402
from dashboard import data as D  # noqa: E402
from dashboard import insights as I  # noqa: E402

PT = "America/Vancouver"
REPO = "https://github.com/mcsare/Live-Gradio-Dashboard-ER-Wait-times"
AUTHORITIES = ["All", "VCH", "PHC", "PHSA", "FHA"]
AUTH_NAME = {"VCH": "Vancouver Coastal", "PHC": "Providence", "PHSA": "PHSA (BC Children's)", "FHA": "Fraser"}
MODELS = ["main", "day1_frozen", "persistence", "seasonal"]
MODEL_CHOICES = [(C.MODEL_LABEL[m], m) for m in MODELS]
HORIZON_CHOICES = [("1 hour", 60), ("2 hours", 120), ("3 hours", 180)]

CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
#hdr h1 {margin-bottom: 2px;}
.status {display:inline-flex; align-items:center; gap:8px; font-size:14px;}
.dot {width:10px; height:10px; border-radius:50%; display:inline-block;}
.disclaimer {font-size:13px; color:#52514e; border-left:3px solid #e34948; padding:4px 10px; margin-top:6px;}
.kpis {display:grid; grid-template-columns:repeat(4,1fr); gap:12px;}
.kpi {border:1px solid #e6e5e1; border-radius:10px; padding:14px 16px; background:#fcfcfb;}
.kpi .label {font-size:13px; color:#52514e;}
.kpi .value {font-size:30px; font-weight:600; color:#0b0b0b; line-height:1.2;}
.kpi .sub {font-size:12px; color:#52514e;}
.insight {display:flex; gap:14px; border:1px solid #e6e5e1; border-radius:10px; padding:14px 16px;
          margin-bottom:10px; background:#fcfcfb;}
.insight .num {font-size:20px; font-weight:600; color:#2a78d6; min-width:22px;}
.insight .finding {font-size:15px; color:#0b0b0b; margin-bottom:6px; line-height:1.45;}
.insight .action {font-size:14px; color:#52514e;}
.insight .action span {font-weight:600; color:#0b0b0b; margin-right:4px;}
.table-wrap td, .table-wrap th {font-family: Inter, system-ui, sans-serif !important; font-size: 13px !important;}
@media (max-width: 800px) {.kpis {grid-template-columns:repeat(2,1fr);}}
"""
# Charts are styled for a light background, so always open in light mode
FORCE_LIGHT = """() => { const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u); } }"""


# ---------- header ----------

def header_html() -> str:
    _, t = D.latest()
    age = None if t is None else (pd.Timestamp.now(tz="UTC") - t).total_seconds() / 60
    if age is not None and age < 30:
        colour, state = "#0ca30c", "Live"
    elif age is not None:
        colour, state = "#eda100", "Data paused: the newest reading is over 30 minutes old"
    else:
        colour, state = "#b9b8b2", "No data yet"
    mode = "" if os.getenv("DATABASE_URL") else " · offline sample data"
    return (f'<span class="status"><span class="dot" style="background:{colour}"></span>'
            f'<b>{state}</b> · Last updated {D.fmt_pt(t, with_day=True)}{mode}</span>')


# ---------- tab 1: live now ----------

def _filter_auth(df: pd.DataFrame, auth: str) -> pd.DataFrame:
    return df if auth == "All" or df.empty else df[df["authority"] == auth]


def _latest_forecasts() -> pd.DataFrame:
    p = D.predictions()
    if p.empty:
        return p
    p = p[p["made_at"] == p["made_at"].max()]
    main = p[p["model_name"] == "main"]
    p = main if len(main) else p[p["model_name"] == "persistence"]
    names = D.facilities().set_index("facility_id")
    return p.assign(name=p["entity_id"].map(names["name"]), authority=p["entity_id"].map(names["authority"]))


def live_view(auth: str, threshold: int):
    latest, t = D.latest()
    latest = _filter_auth(latest, auth)
    posted = latest.dropna(subset=["wait_min"])
    fc = _filter_auth(_latest_forecasts(), auth)

    if len(posted):
        top = posted.loc[posted["wait_min"].idxmax()]
        longest, longest_sub = f"{top['wait_min']:.0f} min", top["name"]
    else:
        longest, longest_sub = "n/a", "no site posting a wait"
    over_now = int((posted["wait_min"] >= threshold).sum())
    over_next = fc[fc["predicted"] >= threshold]["entity_id"].nunique() if len(fc) else 0
    tiles = [
        ("Longest wait now", longest, longest_sub),
        ("Sites posting a wait", f"{len(posted)} of {len(latest)}", "others show 'unavailable'"),
        (f"At or over {threshold} min now", str(over_now), "sites"),
        (f"Forecast at or over {threshold} min", str(over_next), "sites, within 3 hours"),
    ]
    kpis = '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="label">{a}</div><div class="value">{b}</div><div class="sub">{c}</div></div>'
        for a, b, c in tiles) + "</div>"

    if len(fc):
        tab = fc.pivot_table(index="name", columns="horizon_min", values="predicted").round()
        tab.columns = [f"+{c // 60} h" for c in tab.columns]
        now = posted.set_index("name")["wait_min"]
        tab.insert(0, "Now", tab.index.map(now))
        last_col = tab.columns[-1]
        tab["Trend"] = (tab[last_col] - tab["Now"]).apply(
            lambda d: "rising" if d >= 15 else ("falling" if d <= -15 else "steady") if pd.notna(d) else "")
        tab = tab.sort_values("Now", ascending=False).reset_index().rename(columns={"name": "Site"})
        tab = tab.fillna("").astype({c: str for c in tab.columns if c not in ("Site", "Trend")})
        tab = tab.replace(r"\.0$", "", regex=True)
    else:
        tab = pd.DataFrame({"Site": [], "Now": [], "+1 h": [], "+2 h": [], "+3 h": [], "Trend": []})
    return header_html(), kpis, C.live_map(latest, threshold), tab


# ---------- tab 2: explore ----------

def explore_view(site, start, end, h0, h1, days):
    w = D.waits()
    if w.empty:
        e = C.empty("No data yet")
        return e, e, e
    t = w["observed_at"].dt.tz_convert(PT)
    keep = pd.Series(True, index=w.index)
    if start:
        keep &= t.dt.strftime("%Y-%m-%d") >= start
    if end:
        keep &= t.dt.strftime("%Y-%m-%d") <= end
    h0, h1 = int(h0 or 0), int(23 if h1 is None else h1)
    hr = t.dt.hour
    keep &= hr.between(h0, h1) if h0 <= h1 else (hr >= h0) | (hr <= h1)   # e.g. 22 to 6 wraps midnight
    if days:
        keep &= t.dt.strftime("%a").isin(days)
    w = w[keep]
    ws = w if site == "All sites" else w[w["name"] == site]
    return C.heatmap(ws, site), C.timeseries(ws, site), C.dry_vs_rain(ws, D.weather())


# ---------- tab 3: forecast vs actual ----------

def forecast_view(site, horizon, models, start, end):
    horizon = int(horizon)
    models = models or ["main"]
    s = D.scored()
    p = D.predictions()
    w = D.waits()
    names = D.facilities().set_index("facility_id")["name"]
    p = p.assign(name=p["entity_id"].map(names))

    def in_range(ts):
        d = ts.dt.tz_convert(PT).dt.strftime("%Y-%m-%d")
        ok = pd.Series(True, index=ts.index)
        if start:
            ok &= d >= start
        if end:
            ok &= d <= end
        return ok

    if len(s):
        s = s[in_range(s["target_time"])]
    p = p[(p["horizon_min"] == horizon) & p["model_name"].isin(models)]
    if len(p):
        p = p[in_range(p["target_time"])]
    w = w[in_range(w["observed_at"])] if len(w) else w

    if site == "All sites":
        actual = w.assign(t=w["observed_at"].dt.floor("15min")).groupby("t")["wait_min"].mean().reset_index()
        pp = (p.assign(target_time=p["target_time"].dt.floor("15min"))
              .groupby(["model_name", "target_time"])["predicted"].mean().reset_index())
        s_site, label = s, "All reporting sites (average)"
    else:
        actual = w[w["name"] == site][["observed_at", "wait_min"]].rename(columns={"observed_at": "t"})
        pp = p[p["name"] == site].sort_values("target_time")
        s_site, label = (s[s["name"] == site] if len(s) else s), site
    actual = actual.rename(columns={"wait_min": "actual"}).sort_values("t")

    if len(s_site):
        g = s_site.groupby(["model_name", "horizon_min"])["abs_error"].agg(["mean", "count"])
        rows = []
        for m in MODELS:
            row = {"Model": C.MODEL_LABEL[m]}
            for h in (60, 120, 180):
                if (m, h) in g.index:
                    mae = g.loc[(m, h), "mean"]
                    base = g.loc[("persistence", h), "mean"] if ("persistence", h) in g.index else None
                    row[f"Error {C.HORIZON_LABEL[h]} (min)"] = round(mae, 1)
                    row[f"Skill vs no change {C.HORIZON_LABEL[h]}"] = (
                        f"{1 - mae / base:+.0%}" if base and m != "persistence" else "")
            row["Forecasts scored"] = int(g.loc[m, "count"].sum()) if m in g.index.get_level_values(0) else 0
            rows.append(row)
        table = pd.DataFrame(rows)
    else:
        table = pd.DataFrame({"Model": [C.MODEL_LABEL[m] for m in MODELS], "Forecasts scored": [0] * 4})

    return (C.pred_vs_actual(actual, pp, models, label, horizon), C.rolling_mae(s_site, models, horizon),
            table, C.correction_bars(s_site), C.error_by_hour(s_site, horizon))


# ---------- tab 4: insights ----------

def insights_view(auth, threshold):
    w = _filter_auth(D.waits(), auth)
    latest, _ = D.latest()
    latest = _filter_auth(latest, auth)
    p = D.predictions()
    fac = D.facilities().set_index("facility_id")
    p = p.assign(name=p["entity_id"].map(fac["name"]), authority=p["entity_id"].map(fac["authority"]))
    p = _filter_auth(p, auth)
    s = _filter_auth(D.scored().assign(authority=lambda d: d["entity_id"].map(fac["authority"]))
                     if len(D.scored()) else D.scored(), auth)
    scope = "all sites" if auth == "All" else AUTH_NAME.get(auth, auth)
    items = I.build(w, latest, p, s, D.weather(), int(threshold))
    return f"#### What the data says for {scope}\n", I.to_html(items)


# ---------- tab 5: about ----------

def about_md() -> str:
    c = D.counts()
    w = D.waits()
    span = (f"{D.fmt_pt(w['observed_at'].min(), True)} to {D.fmt_pt(w['observed_at'].max(), True)}"
            if len(w) else "no data yet")

    def n(k):
        return f"{c[k]:,}" if c.get(k) is not None else "n/a"

    return f"""
### Sources

| Source | What we use | How often | Access |
|---|---|---|---|
| edwaittimes.ca (Vancouver Coastal, Providence, Fraser Health, PHSA) | Posted wait at each ED and urgent care centre | Every 15 minutes | The public JSON feed the site itself loads |
| Open-Meteo | Hourly temperature, rain and wind for Vancouver | Every 15 minutes | Free API, CC BY 4.0 |
| BCCDC respiratory virus reports | Weekly influenza, RSV and COVID-19 activity level | Weekly | Entered from the Thursday summary |

Data collected: **{span}**.

### How it works

A collector (GitHub Actions, every 15 minutes) writes waits and weather to one Postgres database.
An hourly model job scores past forecasts against what happened, retrains, and writes new forecasts
1, 2 and 3 hours ahead. This dashboard only reads. It refreshes itself every minute.

| Table | Rows | What it holds |
|---|---|---|
| ed_waits | {n('ed_waits')} | One row per site per poll (wait in minutes, or empty if the site hides it) |
| facilities | {n('facilities')} | Site name, ED or urgent care, health authority, location |
| weather_hourly | {n('weather_hourly')} | Temperature, rain and wind by hour |
| resp_activity | {n('resp_activity')} | BCCDC weekly flu, RSV and COVID levels |
| predictions | {n('predictions')} | Every forecast ever made, never edited |
| model_runs | {n('model_runs')} | Each hourly retrain and its validation error |

Tables join on facility (facility_id) and time: weather by the hour, BCCDC by the week.

### Models

| Model | How it predicts |
|---|---|
| Main model | Gradient boosted trees predicting the change from the current wait, retrained every hour, then corrected by its own error at that site over the last 24 hours |
| Day 1 model | The same model trained on the first 24 hours only and never updated, to show what retraining adds |
| No change baseline | The wait later equals the wait now |
| Usual for this hour | Average wait at that site for that hour of day so far |

### Limitations

* Posted waits are the health authorities' own estimates, not measured waits.
* Only the Vancouver Coastal, Providence and BC Children's emergency departments currently post a wait.
  Fraser Health sites and urgent care centres show "unavailable".
* History starts the day the collector started (November 19, 2026), so weekly patterns are thin.
* Missed polls are left as gaps, never filled in.

Code and setup: [{REPO.split('github.com/')[1]}]({REPO})
"""


def refresh_choices():
    sites = ["All sites"] + D.site_choices()
    dates = D.date_choices()
    first, last = (dates[0], dates[-1]) if dates else (None, None)
    return (gr.update(choices=sites), gr.update(choices=sites),
            gr.update(choices=dates, value=first), gr.update(choices=dates, value=last),
            gr.update(choices=dates, value=first), gr.update(choices=dates, value=last))


# ---------- layout ----------

def build_app() -> gr.Blocks:
    eng = get_engine()
    if eng.dialect.name == "sqlite":
        metadata.create_all(eng)  # offline mode with no sample.db yet: empty tables, "No data yet" screens
    sites = ["All sites"] + D.site_choices()
    dates = D.date_choices()
    default_site = sites[1] if len(sites) > 1 else "All sites"

    with gr.Blocks(title="ER Pulse: Metro Vancouver ER wait forecasts") as demo:
        with gr.Column(elem_id="hdr"):
            gr.Markdown(
                "# ER Pulse: Metro Vancouver emergency wait forecasts\n"
                "**For ED operations and patient flow managers** deciding when to add staff and when to start "
                "\"consider an urgent care centre\" messaging. Live posted waits, forecasts 1 to 3 hours ahead, "
                "and how accurate those forecasts have been.")
            status = gr.HTML(header_html())
            gr.HTML('<div class="disclaimer">Wait times are estimates for general information only. '
                    'Never delay emergency care because of a posted wait. <b>In an emergency, call 911.</b></div>')

        with gr.Tabs():
            with gr.Tab("Live now"):
                with gr.Row():
                    auth = gr.Dropdown(AUTHORITIES, value="All", label="Health authority", scale=1)
                    thr = gr.Slider(60, 360, value=180, step=30, label="Alert threshold (minutes)", scale=2)
                kpis = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=5):
                        live_map = gr.Plot(label="Current posted wait by site")
                    with gr.Column(scale=4):
                        gr.Markdown("**Next 3 hours** (main model, minutes)")
                        fc_table = gr.Dataframe(interactive=False, wrap=True, max_height=440,
                                                column_widths=["38%", "11%", "11%", "11%", "11%", "18%"])

            with gr.Tab("Explore"):
                with gr.Row():
                    ex_site = gr.Dropdown(sites, value="All sites", label="Site")
                    ex_start = gr.Dropdown(dates, value=dates[0] if dates else None, label="From")
                    ex_end = gr.Dropdown(dates, value=dates[-1] if dates else None, label="To")
                with gr.Row():
                    ex_h0 = gr.Slider(0, 23, value=0, step=1, label="From hour (Pacific, 0 = midnight)")
                    ex_h1 = gr.Slider(0, 23, value=23, step=1, label="To hour")
                    ex_days = gr.CheckboxGroup(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                               value=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                               label="Days of week")
                ex_heat = gr.Plot()
                with gr.Row():
                    ex_ts = gr.Plot()
                    ex_rain = gr.Plot()

            with gr.Tab("Forecast vs actual"):
                gr.Markdown("Every forecast is saved when it is made and checked against the wait that was "
                            "actually posted. The model retrains every hour, so its error should fall as data "
                            "builds up. The day 1 model never retrains, so the gap between the two is the "
                            "improvement.")
                with gr.Row():
                    fv_site = gr.Dropdown(sites, value=default_site, label="Site")
                    fv_h = gr.Radio(HORIZON_CHOICES, value=120, label="Forecast horizon")
                    fv_models = gr.CheckboxGroup(MODEL_CHOICES, value=["main", "day1_frozen", "persistence"],
                                                 label="Models")
                with gr.Row():
                    fv_start = gr.Dropdown(dates, value=dates[0] if dates else None, label="From")
                    fv_end = gr.Dropdown(dates, value=dates[-1] if dates else None, label="To")
                fv_main = gr.Plot()
                fv_mae = gr.Plot()
                gr.Markdown("**Error and skill by horizon** (skill = how much smaller the error is than "
                            "assuming no change; above 0% means the model helps)")
                fv_table = gr.Dataframe(interactive=False, wrap=True)
                with gr.Row():
                    fv_corr = gr.Plot()
                    fv_hour = gr.Plot()

            with gr.Tab("Insights and actions"):
                with gr.Row():
                    in_auth = gr.Dropdown(AUTHORITIES, value="All", label="Health authority")
                    in_thr = gr.Slider(60, 360, value=180, step=30, label="Alert threshold (minutes)")
                in_title = gr.Markdown()
                in_cards = gr.HTML()

            with gr.Tab("About the data"):
                about = gr.Markdown(about_md())

        timer = gr.Timer(60)
        live_out = [status, kpis, live_map, fc_table]
        ex_in = [ex_site, ex_start, ex_end, ex_h0, ex_h1, ex_days]
        ex_out = [ex_heat, ex_ts, ex_rain]
        fv_in = [fv_site, fv_h, fv_models, fv_start, fv_end]
        fv_out = [fv_main, fv_mae, fv_table, fv_corr, fv_hour]

        for c in (auth, thr):
            c.change(live_view, [auth, thr], live_out)
        for c in ex_in:
            c.change(explore_view, ex_in, ex_out)
        for c in fv_in:
            c.change(forecast_view, fv_in, fv_out)
        for c in (in_auth, in_thr):
            c.change(insights_view, [in_auth, in_thr], [in_title, in_cards])

        timer.tick(live_view, [auth, thr], live_out)
        timer.tick(insights_view, [in_auth, in_thr], [in_title, in_cards])
        demo.load(live_view, [auth, thr], live_out)
        demo.load(explore_view, ex_in, ex_out)
        demo.load(forecast_view, fv_in, fv_out)
        demo.load(insights_view, [in_auth, in_thr], [in_title, in_cards])
        demo.load(about_md, None, about)
        demo.load(refresh_choices, None, [ex_site, fv_site, ex_start, ex_end, fv_start, fv_end])
    return demo


demo = build_app()

if __name__ == "__main__":
    # Render (and most hosts) pass the port in $PORT and need the app to listen on all interfaces
    on_host = "PORT" in os.environ
    demo.launch(css=CSS, js=FORCE_LIGHT, theme=gr.themes.Default(primary_hue="blue"),
                server_name="0.0.0.0" if on_host else None,
                server_port=int(os.environ["PORT"]) if on_host else None)
