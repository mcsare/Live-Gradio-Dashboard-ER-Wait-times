"""Collector entry point. GitHub Actions runs this every 15 minutes.

  python collector/run_loop.py                 one poll, writes to DATABASE_URL
  python collector/run_loop.py --local         one poll, writes to data/local.db
  python collector/run_loop.py --inspect       also print the field names the source sends
  python collector/run_loop.py --from-file x   parse a saved JSON response instead of fetching

Exits with code 1 if any poll fails or returns no wait times, so GitHub
emails the repo owner and the gap is visible in the Actions tab.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os  # noqa: E402

from db import (  # noqa: E402
    ed_waits, facilities, get_engine, insert_ignore, metadata, poll_log, raw_polls, weather_hourly,
)
from collector import sources  # noqa: E402

FACILITY_CSV = ROOT / "data" / "static" / "facilities.csv"


def load_facility_info() -> dict[str, dict]:
    if not FACILITY_CSV.exists():
        return {}
    with open(FACILITY_CSV, newline="", encoding="utf-8") as f:
        return {r["facility_id"]: r for r in csv.DictReader(f)}


def log(conn, source, observed_at, ok, n=None, n_wait=None, error=None):
    conn.execute(poll_log.insert().values(
        polled_at=observed_at, source=source, ok=ok, n_records=n, n_with_wait=n_wait,
        error=(error or "")[:2000] or None))


def poll_waits(engine, info, from_file=None, inspect=False) -> bool:
    observed_at = sources.utcnow()
    try:
        if from_file:
            records = json.loads(Path(from_file).read_text(encoding="utf-8"))
        else:
            records = sources.fetch_waits_raw()
        if inspect and records:
            print("Fields on first record:", sorted(records[0].keys()))
            print("First record:", json.dumps(records[0], indent=2)[:3000])
        missing = sources.EXPECTED_KEYS - set(records[0].keys()) if records else sources.EXPECTED_KEYS
        if missing:
            print(f"WARNING: expected fields missing from source: {sorted(missing)}")

        fac_rows, wait_rows = sources.parse_waits(records, observed_at)
        for r in fac_rows:
            meta = info.get(r["facility_id"], {})
            r["authority"] = meta.get("authority") or "Unknown"
            mv = meta.get("metro_van")
            r["metro_van"] = None if mv in (None, "") else mv.strip().lower() == "true"
            r["first_seen"] = observed_at
            r["last_seen"] = observed_at
        n_wait = sum(1 for w in wait_rows if w["wait_min"] is not None)

        with engine.begin() as conn:
            # A manual run in the same minute as a scheduled one is a harmless no-op
            insert_ignore(conn, raw_polls, [{
                "polled_at": observed_at, "source": "edwaittimes",
                "payload_gz": gzip.compress(json.dumps(records).encode()),
                "n_records": len(records)}], ["polled_at", "source"])
            insert_ignore(conn, facilities, fac_rows, ["facility_id"],
                          update_cols=["name", "type", "raw_type", "authority", "metro_van",
                                       "audience", "address", "lat", "lon", "last_seen"])
            insert_ignore(conn, ed_waits, wait_rows, ["facility_id", "observed_at"])
            ok = n_wait > 0
            log(conn, "edwaittimes", observed_at, ok, len(fac_rows), n_wait,
                None if ok else "No facility returned a wait time")

        unknown = [r["name"] for r in fac_rows if r["authority"] == "Unknown"]
        print(f"{observed_at:%Y-%m-%d %H:%M} UTC  edwaittimes: {len(fac_rows)} facilities, "
              f"{n_wait} with a wait time")
        if unknown:
            print(f"  New facilities not in facilities.csv (add them): {unknown}")
        return ok
    except Exception as e:  # keep a record of every failure
        traceback.print_exc()
        with engine.begin() as conn:
            log(conn, "edwaittimes", observed_at, False, error=f"{type(e).__name__}: {e}")
        return False


def poll_weather(engine) -> bool:
    observed_at = sources.utcnow()
    try:
        rows = sources.fetch_weather()
        with engine.begin() as conn:
            insert_ignore(conn, weather_hourly, rows, ["time"],
                          update_cols=["temp_c", "precip_mm", "wind_kmh", "fetched_at"])
            log(conn, "open-meteo", observed_at, True, len(rows))
        print(f"{observed_at:%Y-%m-%d %H:%M} UTC  open-meteo: {len(rows)} hourly rows upserted")
        return True
    except Exception as e:
        traceback.print_exc()
        with engine.begin() as conn:
            log(conn, "open-meteo", observed_at, False, error=f"{type(e).__name__}: {e}")
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--polls", type=int, default=1)
    p.add_argument("--sleep", type=int, default=60)
    p.add_argument("--local", action="store_true", help="write to data/local.db instead of DATABASE_URL")
    p.add_argument("--from-file", help="parse a saved JSON response (testing)")
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--no-weather", action="store_true")
    a = p.parse_args()

    if not a.local and not os.getenv("DATABASE_URL"):
        print("DATABASE_URL is not set. Add it to .env (or repo secrets), or use --local.")
        return 2

    engine = get_engine(local=a.local)
    metadata.create_all(engine)  # safe to run every time
    info = load_facility_info()

    all_ok = True
    for i in range(a.polls):
        all_ok &= poll_waits(engine, info, a.from_file, a.inspect and i == 0)
        if i < a.polls - 1:
            time.sleep(a.sleep)
    if not a.no_weather and not a.from_file:
        all_ok &= poll_weather(engine)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
