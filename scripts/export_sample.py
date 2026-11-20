"""Copy the Postgres database into data/sample.db (SQLite) for the offline fallback.

  python scripts/export_sample.py            all data
  python scripts/export_sample.py --days 5   only the last 5 days of waits and predictions

raw_polls is left out to keep the file small enough for GitHub.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from db import SAMPLE_DB, get_engine, metadata  # noqa: E402

TIME_COL = {"ed_waits": "observed_at", "predictions": "made_at", "poll_log": "polled_at",
            "weather_hourly": "time"}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int)
    a = p.parse_args()
    src = get_engine()
    if src.dialect.name != "postgresql":
        sys.exit("Set DATABASE_URL to the Postgres database first.")
    if SAMPLE_DB.exists():
        SAMPLE_DB.unlink()
    dst = get_engine(sample=True)
    tables = [t for t in metadata.sorted_tables if t.name != "raw_polls"]
    metadata.create_all(dst, tables=tables)
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=a.days) if a.days else None
    with src.connect() as s, dst.begin() as d:
        for t in tables:
            q = select(t)
            if cutoff is not None and t.name in TIME_COL:
                q = q.where(t.c[TIME_COL[t.name]] >= cutoff)
            rows = [dict(r._mapping) for r in s.execute(q)]
            for i in range(0, len(rows), 5000):
                d.execute(t.insert(), rows[i:i + 5000])
            print(f"{t.name:15s} {len(rows):>8,} rows")
    print(f"Wrote {SAMPLE_DB} ({SAMPLE_DB.stat().st_size / 1e6:.1f} MB)")
