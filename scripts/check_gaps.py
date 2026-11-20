"""Daily health check. Run it once a day while data is being collected.

  python scripts/check_gaps.py [--local] [--hours 48]

Shows row counts, the newest poll, failed polls, hours with fewer than 3
successful polls, and the latest wait at each facility.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from db import get_engine, query  # noqa: E402

PT = "America/Vancouver"


def to_pt(s):
    return pd.to_datetime(s, utc=True).dt.tz_convert(PT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local", action="store_true")
    p.add_argument("--hours", type=int, default=48)
    a = p.parse_args()
    eng = get_engine(local=a.local)

    print("Row counts")
    for t in ["facilities", "ed_waits", "raw_polls", "poll_log", "weather_hourly",
              "resp_activity", "predictions", "model_runs"]:
        n = query(f"select count(*) as n from {t}", engine=eng)["n"].iloc[0]
        print(f"  {t:15s} {n:>8,}")

    log = query("select polled_at, source, ok, n_records, n_with_wait, error from poll_log", engine=eng)
    if log.empty:
        print("\nNo polls yet.")
        return
    log["polled_at"] = to_pt(log["polled_at"])
    log["ok"] = log["ok"].astype(bool)  # SQLite returns 0/1
    since = pd.Timestamp.now(tz=PT) - pd.Timedelta(hours=a.hours)
    waits = log[(log.source == "edwaittimes") & (log.polled_at >= since)]

    newest = log[(log.source == "edwaittimes") & log.ok]["polled_at"].max()
    age = (pd.Timestamp.now(tz=PT) - newest).total_seconds() / 60 if pd.notna(newest) else None
    print(f"\nNewest good poll: {newest:%a %b %d %H:%M} PT ({age:.0f} min ago)" if age is not None
          else "\nNo successful polls yet")
    if age is not None and age > 45:
        print("  WARNING: no good poll in the last 45 minutes. Check the Actions tab.")

    bad = waits[~waits.ok]
    print(f"\nFailed polls in the last {a.hours} h: {len(bad)}")
    if len(bad):
        for _, r in bad.tail(10).iterrows():
            print(f"  {r.polled_at:%b %d %H:%M}  {str(r.error)[:110]}")

    good = waits[waits.ok].set_index("polled_at").resample("1h")["ok"].count()
    thin = good[good < 3]
    print(f"\nHours with fewer than 3 good polls (expect 4): {len(thin)} of {len(good)}")
    if len(thin):
        print(thin.tail(15).to_string())

    latest = query("""
        select f.name, f.authority, f.type, w.wait_min, w.observed_at
        from ed_waits w join facilities f on f.facility_id = w.facility_id
        where w.observed_at = (select max(observed_at) from ed_waits)
        order by w.wait_min desc""", engine=eng)
    if not latest.empty:
        print(f"\nLatest poll ({to_pt(latest['observed_at']).max():%H:%M} PT), sites with a wait:")
        print(latest.dropna(subset=["wait_min"])[["name", "authority", "type", "wait_min"]]
              .to_string(index=False))
        print(f"Sites with no posted wait: {latest['wait_min'].isna().sum()}")

    wx = query("select max(time) as t, count(*) as n from weather_hourly", engine=eng)
    if wx["n"].iloc[0]:
        print(f"\nWeather rows: {wx['n'].iloc[0]}, latest hour {pd.to_datetime(wx['t'].iloc[0], utc=True).tz_convert(PT):%b %d %H:%M} PT")


if __name__ == "__main__":
    main()
