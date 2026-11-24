"""Fill data/local.db with simulated waits in the real schema, for testing offline.

  python tests/simulate_data.py .              4 days ending now
  python tests/simulate_data.py . --days 2

Then run the model job over it hour by hour (see README, Testing offline) and open the
dashboard with DATABASE_URL=sqlite:///data/local.db python app.py
"""
import argparse
import datetime as dt
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--days", type=float, default=4)
a = ap.parse_args()
sys.path.insert(0, a.root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from db import ed_waits, get_engine, insert_ignore, metadata, weather_hourly  # noqa: E402
from scripts.load_static import load_all  # noqa: E402

eng = get_engine(local=True)
metadata.drop_all(eng)
metadata.create_all(eng)
load_all(eng)
recs = json.load(open(f"{a.root}/tests/fixture_wait_times.json"))
pub = [r for r in recs if r["status"] == "published"]
rng = np.random.default_rng(1)
end = pd.Timestamp.now(tz="UTC").floor("15min")
times = pd.date_range(end - pd.Timedelta(days=a.days), end, freq="15min")
rain = {}
rows = []
for k, r in enumerate(pub):
    f, posts = r["id"], r.get("showWaitTimes")
    base = 60 + 12 * k
    walk = np.cumsum(rng.normal(0, 4, len(times)))
    for i, t in enumerate(times):
        if rng.random() < 0.03:
            continue                                     # missed polls
        lt = t.tz_convert("America/Vancouver")
        h = lt.hour + lt.minute / 60
        diurnal = 40 * np.cos(2 * np.pi * (h - 17) / 24)  # smooth daily cycle, peak 5 pm
        w = max(5, base + diurnal + walk[i] * 0.5 + rng.normal(0, 8))
        rows.append({"facility_id": f, "observed_at": t.to_pydatetime() + dt.timedelta(minutes=4),
                     "wait_min": int(round(w / 5) * 5) if posts else None, "show_wait": bool(posts),
                     "source_updated_at": None})
hours = pd.date_range(times[0].floor("1h") - pd.Timedelta("1h"), end + pd.Timedelta("1D"), freq="1h")
wet = (np.sin(np.arange(len(hours)) / 9.0) > 0.6)          # rainy spells of several hours
wx = [{"time": t.to_pydatetime(), "temp_c": 7.0 + 3 * np.sin(i / 4), "precip_mm": 1.2 if wet[i] else 0.0,
       "wind_kmh": 10.0, "fetched_at": None} for i, t in enumerate(hours)]
with eng.begin() as c:
    insert_ignore(c, ed_waits, rows, ["facility_id", "observed_at"])
    insert_ignore(c, weather_hourly, wx, ["time"])
print(f"{len(pub)} facilities, {len(rows)} wait rows, {times[0]} to {times[-1]}")
