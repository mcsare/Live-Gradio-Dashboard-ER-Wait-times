"""Fill data/local.db with simulated waits in the real schema, for testing the model job offline.

  python tests/simulate_data.py .
"""
import sys, json, datetime as dt
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
from db import get_engine, metadata, ed_waits, weather_hourly, insert_ignore
from scripts.load_static import load_all
eng = get_engine(local=True)
metadata.drop_all(eng); metadata.create_all(eng); load_all(eng)
recs = json.load(open(f"{sys.argv[1]}/tests/fixture_wait_times.json"))
ids = [r["id"] for r in recs if r.get("showWaitTimes") and r["status"] == "published"]
rng = np.random.default_rng(1)
times = pd.date_range("2026-11-20T03:00Z", "2026-11-22T06:00Z", freq="15min")
rows = []
for k, f in enumerate(ids):
    base = 60 + 15 * k
    walk = np.cumsum(rng.normal(0, 4, len(times)))
    for i, t in enumerate(times):
        if rng.random() < 0.03: continue                     # missed polls
        lt = t.tz_convert("America/Vancouver"); h = lt.hour + lt.minute / 60
        diurnal = 40 * np.cos(2 * np.pi * (h - 17) / 24)   # smooth daily cycle, peak 5 pm
        w = max(5, base + diurnal + walk[i] * 0.5 + rng.normal(0, 8))
        rows.append({"facility_id": f, "observed_at": t.to_pydatetime() + dt.timedelta(minutes=4),
                     "wait_min": int(round(w / 5) * 5), "show_wait": True, "source_updated_at": None})
wx = [{"time": t.to_pydatetime(), "temp_c": 7.0, "precip_mm": float(rng.random() < 0.3), "wind_kmh": 10.0,
       "fetched_at": None} for t in pd.date_range("2026-11-19T00:00Z", "2026-11-23T00:00Z", freq="1h")]
with eng.begin() as c:
    insert_ignore(c, ed_waits, rows, ["facility_id", "observed_at"])
    insert_ignore(c, weather_hourly, wx, ["time"])
print(len(ids), "facilities", len(rows), "wait rows")
