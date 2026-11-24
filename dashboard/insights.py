"""Insights and actions, recomputed from the database on every refresh.

Nothing here is hard coded: each statement is built from the current data and
drops out if there isn't enough data to support it.
Framing rule: never suggest delaying emergency care.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PT = "America/Vancouver"


def _h(h: int) -> str:
    h = int(h) % 24
    return f"{(h % 12) or 12} {'am' if h < 12 else 'pm'}"


def peak_window(w: pd.DataFrame, width: int = 4):
    """Busiest `width` hour window of the day (wraps past midnight). Returns start hour, window mean, day mean."""
    by_hour = w.groupby(w["observed_at"].dt.tz_convert(PT).dt.hour)["wait_min"].mean().reindex(range(24))
    if by_hour.notna().sum() < 12:
        return None
    filled = by_hour.interpolate(limit_direction="both")
    ext = pd.concat([filled, filled.iloc[: width - 1]])
    roll = ext.rolling(width).mean().iloc[width - 1:].reset_index(drop=True)
    start = int(roll.idxmax()) - (width - 1)
    return start % 24, float(roll.max()), float(by_hour.mean()), filled


def build(waits: pd.DataFrame, latest: pd.DataFrame, preds: pd.DataFrame, scored: pd.DataFrame,
          wx: pd.DataFrame, threshold: int) -> list[tuple[str, str]]:
    """Return a list of (finding, action) pairs."""
    out = []
    if waits.empty:
        return [("Not enough data yet.", "Check back once the collector has run for a few hours.")]

    # 1. Peak hours for the busiest sites: when to add staff
    means = waits.groupby("name")["wait_min"].mean().sort_values(ascending=False)
    lines = []
    for site in means.index[:3]:
        pw = peak_window(waits[waits["name"] == site])
        if pw:
            start, peak, day, _ = pw
            lines.append(f"**{site}**: longest waits {_h(start)} to {_h(start + 4)} "
                         f"(average {peak:.0f} min, vs {day:.0f} min across the day)")
    if lines:
        first = peak_window(waits[waits["name"] == means.index[0]])
        out.append(("Each busy site has a clear daily peak.<br>" + "<br>".join(lines),
                    f"Schedule extra physician and nurse coverage to start about an hour before each "
                    f"site's peak (for {means.index[0]}, by {_h(first[0] - 1)})."))

    # 2. Forecast crossing the threshold: when to start redirect messaging
    if not preds.empty:
        made = preds["made_at"].max()
        nxt = preds[(preds["made_at"] == made) & (preds["model_name"] == "main")]
        if nxt.empty:
            nxt = preds[(preds["made_at"] == made) & (preds["model_name"] == "persistence")]
        hot = nxt[nxt["predicted"] >= threshold]
        if len(hot):
            first = hot.sort_values("target_time").groupby("name").first().sort_values("target_time")
            def when(ts):
                return _h(pd.Timestamp(ts).tz_convert(PT).hour) if pd.Timestamp(ts).minute < 30 else \
                    pd.Timestamp(ts).tz_convert(PT).strftime("%I:%M %p").lstrip("0").lower()
            names = [f"{n} (by {when(r.target_time)})" for n, r in first.iterrows()]
            items = ", ".join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "")
            out.append((f"Forecast to reach {threshold} min or more within 3 hours: {items}.",
                        "Start 'consider an urgent care centre for non urgent needs' messaging now, "
                        "before the wait crosses the line, not after."))
        else:
            out.append((f"No site is forecast to reach {threshold} min in the next 3 hours.",
                        "No redirect messaging needed right now. Keep standard staffing."))

    # 3. Spread between sites right now: where public messaging can point
    now = latest.dropna(subset=["wait_min"])
    eds = now[now["type"] == "ED"]
    if len(eds) >= 2:
        hi, lo = eds.loc[eds["wait_min"].idxmax()], eds.loc[eds["wait_min"].idxmin()]
        if hi["wait_min"] - lo["wait_min"] >= 30:
            out.append((f"Right now waits range from {lo['wait_min']:.0f} min at {lo['name']} "
                        f"to {hi['wait_min']:.0f} min at {hi['name']}.",
                        "Public messaging for non urgent needs can point to sites with shorter posted waits. "
                        "Anyone with an emergency should go to the nearest ED or call 911."))

    # 4. Quietest window: when to schedule non urgent work
    pw_all = peak_window(waits)
    if pw_all:
        filled = pw_all[3]
        low = filled.rolling(3).mean()
        end = int(low.idxmin())
        out.append((f"Across all sites waits are lowest around {_h(end - 2)} to {_h(end + 1)} "
                    f"(average {low.min():.0f} min).",
                    "Use this window for handovers, training and non urgent tasks, and avoid scheduling "
                    "staff breaks during the peak."))

    # 5. How far to trust the forecast
    if not scored.empty:
        g = scored.groupby(["model_name", "horizon_min"])["abs_error"].mean()
        if ("main", 120) in g and ("persistence", 120) in g:
            m, p = g[("main", 120)], g[("persistence", 120)]
            skill = 1 - m / p if p else np.nan
            n = int(((scored["model_name"] == "main") & (scored["horizon_min"] == 120)).sum())
            if skill > 0:
                out.append((f"The 2 hour forecast is off by {m:.0f} min on average, {skill:.0%} better than "
                            f"assuming the wait stays the same ({n:,} forecasts checked).",
                            "Use the 2 hour forecast when deciding whether to call in extra staff."))
            else:
                out.append((f"The 2 hour forecast (off by {m:.0f} min on average) is not yet beating "
                            f"'no change' ({p:.0f} min), based on {n:,} forecasts.",
                            "Treat the forecast as a guide only until it pulls ahead. "
                            "The Forecast vs actual tab shows whether that is happening."))

    # 6. Rain
    if not wx.empty:
        d = waits.assign(hour=waits["observed_at"].dt.floor("1h")).merge(
            wx.rename(columns={"time": "hour"}), on="hour")
        rainy = d[d["precip_mm"] > 0.1]
        if rainy["hour"].nunique() >= 6:
            diff = rainy["wait_min"].mean() - d[d["precip_mm"] <= 0.1]["wait_min"].mean()
            if abs(diff) >= 10:
                out.append((f"Waits average {abs(diff):.0f} min {'longer' if diff > 0 else 'shorter'} "
                            f"in rainy hours ({rainy['hour'].nunique()} rainy hours so far).",
                            "Check the weather forecast when planning next day staffing."
                            if diff > 0 else "Rain is not a reason to add staff on its own."))
    return out[:5]


def to_html(items: list[tuple[str, str]]) -> str:
    cards = []
    for i, (finding, action) in enumerate(items, 1):
        cards.append(
            f'<div class="insight"><div class="num">{i}</div><div>'
            f'<div class="finding">{_md_bold(finding)}</div>'
            f'<div class="action"><span>Action</span> {action}</div></div></div>')
    return "".join(cards) or "<p>Not enough data yet.</p>"


def _md_bold(s: str) -> str:
    parts = s.split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))
