"""Load data/static/facilities.csv and resp_activity.csv into the database.

Run again whenever you edit either CSV (for example each Thursday after
adding the new BCCDC week to resp_activity.csv).

  python scripts/load_static.py [--local]
"""
import csv
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from db import facilities, get_engine, insert_ignore, metadata, resp_activity  # noqa: E402

STATIC = ROOT / "data" / "static"


def _num(v):
    return float(v) if v not in (None, "") else None


def load_all(engine):
    metadata.create_all(engine)
    with open(STATIC / "facilities.csv", newline="", encoding="utf-8") as f:
        fac = [{
            "facility_id": r["facility_id"], "name": r["name"], "type": r["type"],
            "authority": r["authority"], "metro_van": r["metro_van"].lower() == "true",
            "lat": _num(r["lat"]), "lon": _num(r["lon"]),
        } for r in csv.DictReader(f)]
    with open(STATIC / "resp_activity.csv", newline="", encoding="utf-8") as f:
        resp = [{
            "week_start": dt.date.fromisoformat(r["week_start"]),
            "week_end": dt.date.fromisoformat(r["week_end"]),
            "epi_week": int(r["epi_week"]),
            "flu_level": _num(r["flu_level"]), "rsv_level": _num(r["rsv_level"]),
            "covid_level": _num(r["covid_level"]),
            "summary": r["summary"], "source_url": r["source_url"],
        } for r in csv.DictReader(f)]
    with engine.begin() as conn:
        # The collector fills name, coordinates and type from the live feed.
        # The CSV is the source of truth for authority and the Metro Vancouver flag.
        insert_ignore(conn, facilities, fac, ["facility_id"], update_cols=["authority", "metro_van"])
        insert_ignore(conn, resp_activity, resp, ["week_start"],
                      update_cols=["week_end", "epi_week", "flu_level", "rsv_level",
                                   "covid_level", "summary", "source_url"])
    print(f"Loaded {len(fac)} facilities and {len(resp)} BCCDC weeks")


if __name__ == "__main__":
    load_all(get_engine(local="--local" in sys.argv))
