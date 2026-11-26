"""Shared data loading, features and scoring for the model job and the dashboard.

Times are UTC inside the database and in these frames. Calendar features use Pacific time.
"""
from __future__ import annotations

import pandas as pd

from db import query

PT = "America/Vancouver"
SLOT = "15min"
HORIZONS = [60, 120, 180]
TOL = pd.Timedelta("15min")      # how far an observation may be from the time we want
# BC statutory holidays inside the project window and just after
HOLIDAYS = {pd.Timestamp(d).date() for d in ["2026-11-11", "2026-12-25", "2026-12-26", "2027-01-01"]}

FEATURES = ["facility_code", "authority_code", "is_ed", "horizon_min",
            "wait_now", "wait_1h_ago", "delta_1h", "mean_3h", "same_time_yesterday",
            "hour_now", "hour_target", "usual_at_target_hour", "off_day",
            "temp_c", "precip_mm", "flu_level"]
CATEGORICAL = ["facility_code", "authority_code"]
# usual_at_target_hour: the site's average wait at the target hour of day, using only readings
# up to the anchor time (so training rows never see the future). Helps with the evening peaks.
# weekday_target is computed but left out of FEATURES: with under a week of data it
# identifies the exact date and lets the model memorise instead of generalise.


def _utc(s):
    return pd.to_datetime(s, utc=True)


def load_frames(engine, until=None):
    """Return waits (15 min grid), facilities, weather and flu level frames."""
    fac = query("select facility_id, name, type, authority from facilities order by facility_id",
                engine=engine)
    fac["facility_code"] = range(len(fac))              # stable: ordered by id
    auth = sorted(fac["authority"].fillna("Unknown").unique())
    fac["authority_code"] = fac["authority"].fillna("Unknown").map({a: i for i, a in enumerate(auth)})
    fac["is_ed"] = (fac["type"] == "ED").astype(int)

    w = query("select facility_id, observed_at, wait_min from ed_waits where wait_min is not null",
              engine=engine)
    w["observed_at"] = _utc(w["observed_at"])
    if until is not None:
        w = w[w["observed_at"] <= until]
    # One value per facility per 15 minute slot (the latest in the slot)
    w["slot"] = w["observed_at"].dt.floor(SLOT)
    w = (w.sort_values("observed_at").groupby(["facility_id", "slot"], as_index=False)
         .last()[["facility_id", "slot", "wait_min"]])

    wx = query("select time, temp_c, precip_mm from weather_hourly", engine=engine)
    wx["time"] = _utc(wx["time"])
    resp = query("select week_start, flu_level from resp_activity", engine=engine)
    resp["week_start"] = _utc(resp["week_start"])
    return w, fac, wx, resp


def _asof(left, right, left_on, value_name, tol=TOL):
    """For each left row, the facility's wait nearest to left[left_on] within tol."""
    r = right.rename(columns={"slot": "_t", "wait_min": value_name})[["facility_id", "_t", value_name]]
    out = pd.merge_asof(left.sort_values(left_on), r.sort_values("_t"), left_on=left_on,
                        right_on="_t", by="facility_id", direction="nearest", tolerance=tol)
    return out.drop(columns="_t")


def _usual_at_hour(rows, waits):
    """Add usual_at_target_hour: the facility's mean wait at hour_target (Pacific), from readings
    at or before the anchor time t only. If that hour has not come up yet (early in collection),
    fall back to the facility's mean wait over all hours so far."""
    w = waits.assign(hour_target=waits["slot"].dt.tz_convert(PT).dt.hour.astype("int64"))
    w = w.sort_values("slot")
    g = w.groupby(["facility_id", "hour_target"])["wait_min"]
    w["usual_at_target_hour"] = g.cumsum() / (g.cumcount() + 1)
    r = w[["facility_id", "hour_target", "slot", "usual_at_target_hour"]].rename(columns={"slot": "_t"})
    rows = rows.assign(hour_target=rows["hour_target"].astype("int64"))
    out = pd.merge_asof(rows.sort_values("t"), r, left_on="t", right_on="_t",
                        by=["facility_id", "hour_target"], direction="backward").drop(columns="_t")
    gf = w.groupby("facility_id")["wait_min"]
    w["_fac_mean"] = gf.cumsum() / (gf.cumcount() + 1)
    f = w[["facility_id", "slot", "_fac_mean"]].rename(columns={"slot": "_t"})
    out = pd.merge_asof(out, f, left_on="t", right_on="_t", by="facility_id", direction="backward")
    out["usual_at_target_hour"] = out["usual_at_target_hour"].fillna(out["_fac_mean"])
    return out.drop(columns=["_t", "_fac_mean"])


def build_rows(anchors, waits, fac, wx, resp, horizons=HORIZONS, with_target=True):
    """anchors: facility_id, t (a slot with a known wait_now). Returns one row per anchor and horizon."""
    a = anchors.merge(waits.rename(columns={"slot": "t", "wait_min": "wait_now"}),
                      on=["facility_id", "t"], how="inner")
    a["t_1h"] = a["t"] - pd.Timedelta("1h")
    a = _asof(a, waits, "t_1h", "wait_1h_ago")
    a["t_24h"] = a["t"] - pd.Timedelta("24h")
    a = _asof(a, waits, "t_24h", "same_time_yesterday", tol=pd.Timedelta("30min"))
    a["delta_1h"] = a["wait_now"] - a["wait_1h_ago"]

    # Mean of the last 3 hours, per facility, time based
    roll = (waits.set_index("slot").groupby("facility_id")["wait_min"]
            .rolling("3h").mean().rename("mean_3h").reset_index()
            .rename(columns={"slot": "t"}))
    a = a.merge(roll, on=["facility_id", "t"], how="left")

    rows = pd.concat([a.assign(horizon_min=h) for h in horizons], ignore_index=True)
    rows["target_time"] = rows["t"] + pd.to_timedelta(rows["horizon_min"], unit="min")
    tp = rows["target_time"].dt.tz_convert(PT)
    rows["hour_target"] = tp.dt.hour
    rows["weekday_target"] = tp.dt.weekday
    rows["off_day"] = ((tp.dt.weekday >= 5) | tp.dt.date.isin(HOLIDAYS)).astype(int)
    rows["hour_now"] = rows["t"].dt.tz_convert(PT).dt.hour
    rows = _usual_at_hour(rows, waits)

    rows["wx_hour"] = rows["target_time"].dt.floor("1h")
    rows = rows.merge(wx.rename(columns={"time": "wx_hour"}), on="wx_hour", how="left")
    if len(resp):
        rows = pd.merge_asof(rows.sort_values("target_time"), resp.sort_values("week_start"),
                             left_on="target_time", right_on="week_start", direction="backward")
    else:
        rows["flu_level"] = float("nan")
    rows = rows.merge(fac[["facility_id", "facility_code", "authority_code", "is_ed"]],
                      on="facility_id", how="left")
    if with_target:
        rows = _asof(rows, waits, "target_time", "actual")
    return rows


def training_rows(waits, fac, wx, resp, end=None):
    """All rows whose target is known, using only data up to `end`."""
    if end is not None:
        waits = waits[waits["slot"] <= end]
    rows = build_rows(waits[["facility_id", "slot"]].rename(columns={"slot": "t"}),
                      waits, fac, wx, resp)
    rows = rows.dropna(subset=["actual"])
    if end is not None:
        rows = rows[rows["target_time"] <= end]
    return rows


def seasonal_table(waits):
    """Mean wait per facility and Pacific hour of day, plus a per facility fallback."""
    w = waits.assign(hour=waits["slot"].dt.tz_convert(PT).dt.hour)
    by_hour = w.groupby(["facility_id", "hour"])["wait_min"].mean().rename("seasonal").reset_index()
    overall = w.groupby("facility_id")["wait_min"].mean().rename("fac_mean").reset_index()
    return by_hour, overall


def score(preds: pd.DataFrame, waits_raw: pd.DataFrame) -> pd.DataFrame:
    """Attach the actual wait observed nearest each target_time (within 15 min).

    preds: predictions rows (entity_id, target_time, predicted, ...).
    waits_raw: facility_id, observed_at, wait_min (wait_min not null).
    Rows without an actual yet are dropped.
    """
    p = preds.copy()
    p["target_time"] = _utc(p["target_time"])
    w = waits_raw.rename(columns={"facility_id": "entity_id", "observed_at": "_t", "wait_min": "actual"})
    w["_t"] = _utc(w["_t"])
    out = pd.merge_asof(p.sort_values("target_time"), w.sort_values("_t")[["entity_id", "_t", "actual"]],
                        left_on="target_time", right_on="_t", by="entity_id",
                        direction="nearest", tolerance=TOL).drop(columns="_t")
    out = out.dropna(subset=["actual"])
    out["error"] = out["predicted"] - out["actual"]
    out["abs_error"] = out["error"].abs()
    return out
