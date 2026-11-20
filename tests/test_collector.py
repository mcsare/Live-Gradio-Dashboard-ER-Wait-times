"""Offline tests. Run: python -m pytest tests -q

fixture_wait_times.json mimics the edwaittimes.ca /api/wait-times format,
built from a real snapshot of Nov 1 2026.
"""
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collector import sources  # noqa: E402

FIX = Path(__file__).parent


def test_parse_waits():
    recs = json.loads((FIX / "fixture_wait_times.json").read_text())
    t = dt.datetime(2026, 11, 20, 18, 0, tzinfo=dt.timezone.utc)
    fac, waits = sources.parse_waits(recs, t)
    assert len(fac) == 41                      # draft site skipped
    assert {f["type"] for f in fac} == {"ED", "UPCC"}
    vgh = next(w for w, f in zip(waits, fac) if f["name"] == "Vancouver General Hospital")
    assert vgh["wait_min"] == 180 and vgh["show_wait"]
    hidden = [w for w in waits if not w["show_wait"]]
    assert hidden and all(w["wait_min"] is None for w in hidden)   # hidden waits stored as null
    assert vgh["source_updated_at"].tzinfo is not None


def test_parse_weather():
    js = json.loads((FIX / "fixture_weather.json").read_text())
    rows = sources.parse_weather(js)
    assert len(rows) == 24 and rows[0]["time"].tzinfo == dt.timezone.utc
