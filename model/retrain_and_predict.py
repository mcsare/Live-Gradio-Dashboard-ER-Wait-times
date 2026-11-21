"""Hourly model job: score, retrain, correct, predict.

  python model/retrain_and_predict.py            uses DATABASE_URL
  python model/retrain_and_predict.py --local    uses data/local.db
  python model/retrain_and_predict.py --now 2026-11-21T18:00Z   pretend it is that time (testing)

Every run writes, for each facility that posted a wait in the last 30 minutes and
each horizon (60, 120, 180 min), one prediction per model:

  persistence   wait later = wait now
  seasonal      mean wait for that facility at that hour of day, data so far
  main          gradient boosted trees (predicting the change from the current wait),
                retrained every run on all data so far,
                then bias corrected (predicted_raw keeps the uncorrected value)
  day1_frozen   same model trained only on the first 24 hours, never updated

Predictions are append only. They are scored later against what actually happened
(see model/features.py score()).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from db import get_engine, insert_ignore, metadata, model_runs, predictions, query  # noqa: E402
from model import features as F  # noqa: E402

MIN_TRAIN_ROWS = 300
DAY1_VERSION = "day1-frozen"
PARAMS = dict(loss="absolute_error", learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
              min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, random_state=0)


class ChangeModel:
    """Predicts the change from the current wait, then adds it back.
    With little data this keeps the model close to persistence unless it has a reason not to be."""

    def __init__(self):
        cat_mask = [c in F.CATEGORICAL for c in F.FEATURES]
        self.m = HistGradientBoostingRegressor(categorical_features=cat_mask, **PARAMS)

    def fit(self, rows):
        self.m.fit(rows[F.FEATURES], rows["actual"] - rows["wait_now"])
        return self

    def predict(self, X):
        return X["wait_now"].to_numpy() + self.m.predict(X[F.FEATURES])


def fit(rows: pd.DataFrame) -> ChangeModel:
    return ChangeModel().fit(rows)


def time_split_mae(rows: pd.DataFrame) -> float | None:
    """Train on the first 80% of anchor times, score on the last 20%."""
    cut = rows["t"].quantile(0.8)
    # Training targets must end before validation starts, or the score is too optimistic
    tr, va = rows[rows["target_time"] <= cut], rows[rows["t"] > cut]
    if len(tr) < MIN_TRAIN_ROWS or len(va) < 30:
        return None
    m = fit(tr)
    return float(np.mean(np.abs(m.predict(va) - va["actual"])))


def recent_bias(engine, now) -> dict[str, float]:
    """Mean error (raw prediction minus actual) of the main model per facility over the last 24 h."""
    p = query("""select entity_id, target_time, predicted_raw as predicted from predictions
                 where model_name = 'main' and target_time > :a and target_time <= :b""",
              {"a": now - dt.timedelta(hours=24), "b": now}, engine=engine)
    if p.empty:
        return {}
    w = query("select facility_id, observed_at, wait_min from ed_waits "
              "where wait_min is not null and observed_at > :a",
              {"a": now - dt.timedelta(hours=25)}, engine=engine)
    s = F.score(p, w)
    g = s.groupby("entity_id")["error"].agg(["mean", "count"])
    return g[g["count"] >= 3]["mean"].to_dict()   # need a few scored points before correcting


def already_ran(engine, now) -> bool:
    p = query("select max(made_at) as m from predictions where model_name = 'main'", engine=engine)
    last = p["m"].iloc[0]
    return pd.notna(last) and (pd.Timestamp(now) - pd.to_datetime(last, utc=True)) < pd.Timedelta("40min")


def log_run(engine, version, name, rows, mae, now, extra=None):
    with engine.begin() as conn:
        insert_ignore(conn, model_runs, [{
            "model_version": version, "model_name": name, "trained_at": now,
            "train_rows": int(len(rows)),
            "train_start": rows["t"].min().to_pydatetime(), "train_end": rows["target_time"].max().to_pydatetime(),
            "val_mae": mae, "params_json": json.dumps({**PARAMS, **(extra or {})}),
        }], ["model_version"])


def print_scoreboard(engine, now):
    p = query("select entity_id, target_time, horizon_min, model_name, predicted from predictions "
              "where target_time <= :n", {"n": now}, engine=engine)
    if p.empty:
        print("No predictions have come due yet.")
        return
    w = query("select facility_id, observed_at, wait_min from ed_waits where wait_min is not null",
              engine=engine)
    s = F.score(p, w)
    if s.empty:
        print("No predictions scored yet.")
        return
    t = s.pivot_table(index="model_name", columns="horizon_min", values="abs_error", aggfunc="mean").round(1)
    t["scored"] = s.groupby("model_name").size()
    print("MAE in minutes by model and horizon (all scored predictions so far):")
    print(t.to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--now", help="UTC time to act as now, for testing")
    ap.add_argument("--force", action="store_true", help="run even if a run happened in the last 40 min")
    a = ap.parse_args()

    engine = get_engine(local=a.local)
    metadata.create_all(engine)
    now = (pd.Timestamp(a.now) if a.now else pd.Timestamp.now(tz="UTC")).tz_convert("UTC").floor("min")
    now_py = now.to_pydatetime()

    if not a.force and already_ran(engine, now):
        print("A run already happened in the last 40 minutes, skipping. Use --force to override.")
        return 0

    waits, fac, wx, resp = F.load_frames(engine, until=now)
    if waits.empty:
        print("No wait data yet.")
        return 0

    # 1. Score (printed to the log; the dashboard scores on the fly the same way)
    print_scoreboard(engine, now_py)

    # 2. Retrain the main model on everything so far
    rows = F.training_rows(waits, fac, wx, resp, end=now)
    version = f"main-{now:%Y%m%d%H%M}"
    model = None
    if len(rows) >= MIN_TRAIN_ROWS:
        mae = time_split_mae(rows)
        model = fit(rows)
        log_run(engine, version, "main", rows, mae, now_py)
        print(f"Trained {version} on {len(rows):,} rows; validation MAE "
              f"{'n/a' if mae is None else f'{mae:.1f} min'}")
    else:
        print(f"Only {len(rows)} training rows (need {MIN_TRAIN_ROWS}); main model skipped this hour.")

    # Day 1 model: first 24 hours only. Deterministic, so refitting gives the same frozen model.
    first = waits["slot"].min()
    day1 = None
    if now - first >= pd.Timedelta("24h"):
        d1_rows = F.training_rows(waits, fac, wx, resp, end=first + pd.Timedelta("24h"))
        if len(d1_rows) >= MIN_TRAIN_ROWS:
            day1 = fit(d1_rows)
            exists = query("select count(*) as n from model_runs where model_version = :v",
                           {"v": DAY1_VERSION}, engine=engine)["n"].iloc[0]
            if not exists:
                log_run(engine, DAY1_VERSION, "day1_frozen", d1_rows, None, now_py,
                        {"trained_on": "first 24 hours of collection"})

    # 3. Anchors: each facility's latest wait, if it is fresh
    latest = waits.sort_values("slot").groupby("facility_id").tail(1)
    latest = latest[latest["slot"] >= now - pd.Timedelta("30min")]
    if latest.empty:
        print("No facility posted a wait in the last 30 minutes; no predictions written.")
        return 1
    X = F.build_rows(latest[["facility_id", "slot"]].rename(columns={"slot": "t"}),
                     waits, fac, wx, resp, with_target=False)
    # Targets are measured from now, not from the slot start
    X["target_time"] = now + pd.to_timedelta(X["horizon_min"], unit="min")

    by_hour, overall = F.seasonal_table(waits)
    X["hour"] = X["hour_target"]
    X = X.merge(by_hour, on=["facility_id", "hour"], how="left").merge(overall, on="facility_id", how="left")
    X["seasonal"] = X["seasonal"].fillna(X["fac_mean"]).fillna(X["wait_now"])

    bias = recent_bias(engine, now_py)
    out = []

    def add(name, version_, raw, corrected=None):
        corrected = raw if corrected is None else corrected
        for (_, r), pr, pc in zip(X.iterrows(), raw, corrected):
            out.append({"entity_id": r["facility_id"], "made_at": now_py,
                        "target_time": r["target_time"].to_pydatetime(),
                        "horizon_min": int(r["horizon_min"]),
                        "predicted": float(max(pc, 0)), "predicted_raw": float(max(pr, 0)),
                        "model_version": version_, "model_name": name})

    add("persistence", "persistence", X["wait_now"].to_numpy())
    add("seasonal", "seasonal", X["seasonal"].to_numpy())
    if model is not None:
        raw = model.predict(X)
        b = X["facility_id"].map(bias).fillna(0).to_numpy()
        add("main", version, raw, raw - b)
    if day1 is not None:
        add("day1_frozen", DAY1_VERSION, day1.predict(X))

    with engine.begin() as conn:
        conn.execute(predictions.insert(), out)
    print(f"Wrote {len(out)} predictions for {X['facility_id'].nunique()} facilities "
          f"(bias correction active for {len(bias)} facilities)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
