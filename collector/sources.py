"""Fetch and parse the live sources.

edwaittimes.ca: the site's own front end loads https://edwaittimes.ca/api/wait-times,
a JSON list with one object per facility. Fields we rely on (confirmed Nov 19 2026
against two independent open source parsers of the same feed):

  id, name, status ("published"), type ("upcc" for urgent care, otherwise ED),
  address, latitude, longitude, audience, showWaitTimes,
  waitTime.waitTimeMinutes, waitTime.createdAt

Sites with showWaitTimes = false are stored with wait_min = null.
"""
from __future__ import annotations

import datetime as dt
import os

import requests

WAITS_URL = "https://edwaittimes.ca/api/wait-times"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
# Central Vancouver. One point is enough for regional weather effects.
WEATHER_LAT, WEATHER_LON = 49.2609, -123.1139

EXPECTED_KEYS = {"id", "name", "type", "latitude", "longitude", "showWaitTimes", "waitTime"}


def headers() -> dict:
    contact = os.getenv("CONTACT_EMAIL", "not set")
    return {
        "User-Agent": f"CPSC4820 student project, Langara College (contact: {contact})",
        "Accept": "application/json",
    }


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)


def parse_iso(value) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    except ValueError:
        return None


def fetch_waits_raw() -> list[dict]:
    r = requests.get(WAITS_URL, headers=headers(), timeout=30)
    r.raise_for_status()
    data = r.json()
    # Accept both a bare list and {"locations": [...]} in case the shape changes
    if isinstance(data, dict):
        data = data.get("locations") or data.get("data") or []
    if not isinstance(data, list):
        raise ValueError(f"Unexpected payload type: {type(data).__name__}")
    return data


def parse_waits(records: list[dict], observed_at: dt.datetime) -> tuple[list[dict], list[dict]]:
    """Return (facility rows, wait rows) for published sites."""
    fac_rows, wait_rows = [], []
    for rec in records:
        if not isinstance(rec, dict) or not rec.get("id") or not rec.get("name"):
            continue
        if rec.get("status", "published") != "published":
            continue
        raw_type = str(rec.get("type") or "")
        wt = rec.get("waitTime") or {}
        minutes = wt.get("waitTimeMinutes")
        show = bool(rec.get("showWaitTimes"))
        valid = show and isinstance(minutes, (int, float)) and minutes >= 0
        fac_rows.append({
            "facility_id": str(rec["id"]),
            "name": rec["name"].strip(),
            "type": "UPCC" if raw_type.lower() == "upcc" else "ED",
            "raw_type": raw_type,
            "audience": rec.get("audience"),
            "address": rec.get("address"),
            "lat": rec.get("latitude"),
            "lon": rec.get("longitude"),
        })
        wait_rows.append({
            "facility_id": str(rec["id"]),
            "observed_at": observed_at,
            "wait_min": int(round(minutes)) if valid else None,
            "show_wait": show,
            "source_updated_at": parse_iso(wt.get("createdAt")),
        })
    return fac_rows, wait_rows


def fetch_weather() -> list[dict]:
    """Hourly weather for yesterday, today and tomorrow (UTC). Upserted each run,
    so past hours settle to their final values and future hours are the forecast."""
    params = {
        "latitude": WEATHER_LAT, "longitude": WEATHER_LON,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "past_days": 1, "forecast_days": 2, "timezone": "GMT",
    }
    r = requests.get(WEATHER_URL, params=params, headers=headers(), timeout=30)
    r.raise_for_status()
    return parse_weather(r.json())


def parse_weather(js: dict) -> list[dict]:
    h = js["hourly"]
    fetched = utcnow()
    rows = []
    for i, t in enumerate(h["time"]):
        rows.append({
            "time": dt.datetime.fromisoformat(t).replace(tzinfo=dt.timezone.utc),
            "temp_c": h["temperature_2m"][i],
            "precip_mm": h["precipitation"][i],
            "wind_kmh": h["wind_speed_10m"][i],
            "fetched_at": fetched,
        })
    return rows
