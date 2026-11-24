"""Read only data access for the dashboard, cached for 60 seconds.

Every function returns times as UTC pandas timestamps. Convert to Pacific only for display.
"""
from __future__ import annotations

import time

import pandas as pd

from db import get_engine, query
from model.features import score

PT = "America/Vancouver"
TTL = 60  # seconds; keeps many viewers from hitting the database
_cache: dict[str, tuple[float, object]] = {}


def cached(fn):
    def wrapper():
        hit = _cache.get(fn.__name__)
        if hit and time.time() - hit[0] < TTL:
            return hit[1]
        val = fn()
        _cache[fn.__name__] = (time.time(), val)
        return val
    wrapper.__name__ = fn.__name__
    return wrapper


def _utc(s):
    return pd.to_datetime(s, utc=True)


@cached
def facilities() -> pd.DataFrame:
    return query("select facility_id, name, type, authority, metro_van, lat, lon from facilities",
                 engine=get_engine())


@cached
def waits_all() -> pd.DataFrame:
    """Every poll row, including sites with no posted wait, joined to facility info."""
    w = query("select facility_id, observed_at, wait_min from ed_waits", engine=get_engine())
    w["observed_at"] = _utc(w["observed_at"])
    return w.merge(facilities(), on="facility_id", how="left")


def waits() -> pd.DataFrame:
    """Rows with a posted wait only."""
    w = waits_all()
    return w[w["wait_min"].notna()].copy()


@cached
def predictions() -> pd.DataFrame:
    p = query("select entity_id, made_at, target_time, horizon_min, predicted, predicted_raw, "
              "model_version, model_name from predictions", engine=get_engine())
    p["made_at"] = _utc(p["made_at"])
    p["target_time"] = _utc(p["target_time"])
    return p


@cached
def scored() -> pd.DataFrame:
    p = predictions()
    w = waits()
    if p.empty or w.empty:
        return pd.DataFrame(columns=list(p.columns) + ["actual", "error", "abs_error", "name"])
    s = score(p, w[["facility_id", "observed_at", "wait_min"]])
    # Error of the uncorrected value, for the bias correction chart
    s["abs_error_raw"] = (s["predicted_raw"] - s["actual"]).abs()
    names = facilities().set_index("facility_id")["name"]
    s["name"] = s["entity_id"].map(names)
    return s


@cached
def weather() -> pd.DataFrame:
    wx = query("select time, temp_c, precip_mm from weather_hourly", engine=get_engine())
    wx["time"] = _utc(wx["time"])
    return wx


@cached
def counts() -> dict:
    out = {}
    for t in ["facilities", "ed_waits", "weather_hourly", "resp_activity", "predictions", "model_runs",
              "poll_log"]:
        try:
            out[t] = int(query(f"select count(*) as n from {t}", engine=get_engine())["n"].iloc[0])
        except Exception:
            out[t] = None
    return out


def latest() -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """The most recent poll for every site, and that poll's time."""
    w = waits_all()
    if w.empty:
        return w, None
    t = w["observed_at"].max()
    return w[w["observed_at"] == t].copy(), t


def fmt_pt(ts, with_day=False) -> str:
    if ts is None or pd.isna(ts):
        return "never"
    t = pd.Timestamp(ts).tz_convert(PT)
    s = t.strftime("%a %b %d, %I:%M %p" if with_day else "%I:%M %p").replace(" 0", " ")
    return s.lstrip("0").replace("AM", "am").replace("PM", "pm") + " PT"


def site_choices() -> list[str]:
    """Sites that have posted at least one wait, busiest first."""
    w = waits()
    if w.empty:
        return []
    return w.groupby("name")["wait_min"].mean().sort_values(ascending=False).index.tolist()


def date_choices() -> list[str]:
    w = waits()
    if w.empty:
        return []
    return sorted(w["observed_at"].dt.tz_convert(PT).dt.strftime("%Y-%m-%d").unique().tolist())
