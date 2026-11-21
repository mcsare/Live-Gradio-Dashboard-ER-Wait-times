"""Database access shared by the collector, the model job and the dashboard.

If DATABASE_URL is set we use Postgres (Neon or Supabase).
If it is not set we fall back to data/sample.db (SQLite), so the dashboard runs offline.
All timestamps are stored in UTC.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, Index, Integer, LargeBinary, MetaData,
    String, Table, Text, create_engine, text,
)
from sqlalchemy.engine import Engine

load_dotenv()

ROOT = Path(__file__).resolve().parent
SAMPLE_DB = ROOT / "data" / "sample.db"
LOCAL_DB = ROOT / "data" / "local.db"

metadata = MetaData()

# Static facility list. Filled automatically from each poll, with authority and
# metro flag taken from data/static/facilities.csv.
facilities = Table(
    "facilities", metadata,
    Column("facility_id", String(64), primary_key=True),   # edwaittimes.ca id
    Column("name", String(200), nullable=False),
    Column("type", String(8)),            # ED or UPCC
    Column("raw_type", String(40)),       # type exactly as the source sends it
    Column("authority", String(20)),      # VCH, PHC, PHSA, FHA, Unknown
    Column("metro_van", Boolean),         # inside Metro Vancouver
    Column("audience", String(40)),
    Column("address", String(300)),
    Column("lat", Float),
    Column("lon", Float),
    Column("first_seen", DateTime(timezone=True)),
    Column("last_seen", DateTime(timezone=True)),
)

# One row per facility per poll. wait_min is null when the site hides its wait.
ed_waits = Table(
    "ed_waits", metadata,
    Column("facility_id", String(64), primary_key=True),
    Column("observed_at", DateTime(timezone=True), primary_key=True),  # our poll time, UTC
    Column("wait_min", Integer),
    Column("show_wait", Boolean),
    Column("source_updated_at", DateTime(timezone=True)),  # waitTime.createdAt from the source
)
Index("ix_ed_waits_observed", ed_waits.c.observed_at)

# Full raw response for every poll (gzip), so a format change never loses data.
raw_polls = Table(
    "raw_polls", metadata,
    Column("polled_at", DateTime(timezone=True), primary_key=True),
    Column("source", String(40), primary_key=True),
    Column("payload_gz", LargeBinary),
    Column("n_records", Integer),
)

# One row per collector attempt, success or failure. Used to find gaps.
poll_log = Table(
    "poll_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("polled_at", DateTime(timezone=True), nullable=False),
    Column("source", String(40), nullable=False),
    Column("ok", Boolean, nullable=False),
    Column("n_records", Integer),
    Column("n_with_wait", Integer),
    Column("error", Text),
)

weather_hourly = Table(
    "weather_hourly", metadata,
    Column("time", DateTime(timezone=True), primary_key=True),  # hour start, UTC
    Column("temp_c", Float),
    Column("precip_mm", Float),
    Column("wind_kmh", Float),
    Column("fetched_at", DateTime(timezone=True)),
)

# Weekly BCCDC levels. 1 low, 2 moderate, 3 high, 4 very high (0.5 steps for "low to moderate").
resp_activity = Table(
    "resp_activity", metadata,
    Column("week_start", Date, primary_key=True),
    Column("week_end", Date),
    Column("epi_week", Integer),
    Column("flu_level", Float),
    Column("rsv_level", Float),
    Column("covid_level", Float),
    Column("summary", Text),
    Column("source_url", Text),
)

# Created now so the modeller can start writing on day one. Append only.
predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("entity_id", String(64), nullable=False),
    Column("made_at", DateTime(timezone=True), nullable=False),
    Column("target_time", DateTime(timezone=True), nullable=False),
    Column("horizon_min", Integer, nullable=False),
    Column("predicted", Float),        # after bias correction
    Column("predicted_raw", Float),    # before bias correction
    Column("model_version", String(40)),
    Column("model_name", String(40)),  # persistence, seasonal, main, day1_frozen
)
Index("ix_pred_entity_target", predictions.c.entity_id, predictions.c.target_time)

model_runs = Table(
    "model_runs", metadata,
    Column("model_version", String(40), primary_key=True),
    Column("model_name", String(40)),
    Column("trained_at", DateTime(timezone=True)),
    Column("train_rows", Integer),
    Column("train_start", DateTime(timezone=True)),
    Column("train_end", DateTime(timezone=True)),
    Column("val_mae", Float),
    Column("params_json", Text),
)


def _normalise_url(url: str) -> str:
    # Supabase and Neon hand out postgres:// or postgresql:// strings
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url


@functools.lru_cache(maxsize=None)  # reuse one engine (and its connection pool) per process
def get_engine(local: bool = False, sample: bool = False) -> Engine:
    """Postgres when DATABASE_URL is set. Otherwise local.db (local=True) or sample.db."""
    url = os.getenv("DATABASE_URL", "").strip()
    if url and not (local or sample):
        return create_engine(_normalise_url(url), pool_pre_ping=True)
    path = LOCAL_DB if local else SAMPLE_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}")


def insert_ignore(conn, table: Table, rows: list[dict], keys: list[str],
                  update_cols: list[str] | None = None) -> None:
    """Insert rows, skipping (or updating) rows whose key already exists. Works on Postgres and SQLite."""
    if not rows:
        return
    if conn.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    stmt = insert(table)
    if update_cols:
        stmt = stmt.on_conflict_do_update(
            index_elements=keys, set_={c: stmt.excluded[c] for c in update_cols})
    else:
        stmt = stmt.on_conflict_do_nothing(index_elements=keys)
    conn.execute(stmt, rows)


def query(sql: str, params: dict | None = None, engine: Engine | None = None) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame. Used by the dashboard."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})
