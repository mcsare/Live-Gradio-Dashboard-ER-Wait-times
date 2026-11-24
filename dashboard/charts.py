"""Plotly figures. One look everywhere: thin lines, recessive grid, minutes on axes, Pacific time.

Colours (validated for colour vision deficiency, adjacent pairs):
  each model keeps the same colour on every chart; the actual wait is always near black.
  Magnitude (higher wait = worse) always uses the same one hue blue ramp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PT = "America/Vancouver"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
NO_DATA = "#b9b8b2"

MODEL_LABEL = {"main": "Main model", "day1_frozen": "Day 1 model (frozen)",
               "persistence": "No change baseline", "seasonal": "Usual for this hour"}
MODEL_COLOR = {"main": "#2a78d6", "day1_frozen": "#eb6834",
               "persistence": "#1baf7a", "seasonal": "#eda100"}
# Second encoding so lines never rely on colour alone
MODEL_DASH = {"main": "solid", "day1_frozen": "dash", "persistence": "dot", "seasonal": "dashdot"}
BLUES = [[0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]]
HORIZON_LABEL = {60: "1 hour", 120: "2 hours", 180: "3 hours"}


def style(fig: go.Figure, title: str, y_title: str | None = "Wait (minutes)", height=380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)),
        template="plotly_white", height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=INK_2),
        margin=dict(l=60, r=20, t=60, b=50), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=y_title, rangemode="tozero")
    return fig


def empty(msg: str, height=320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=INK_2), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      margin=dict(l=20, r=20, t=20, b=20))
    return fig


def to_pt(s: pd.Series) -> pd.Series:
    return s.dt.tz_convert(PT).dt.tz_localize(None)


# ---------- Live now ----------

def live_map(latest: pd.DataFrame, threshold: int) -> go.Figure:
    if latest.empty:
        return empty("No data yet")
    d = latest.dropna(subset=["lat", "lon"]).copy()
    has = d[d["wait_min"].notna()]
    none = d[d["wait_min"].isna()]
    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=none["lat"], lon=none["lon"], mode="markers", name="No posted wait",
        marker=dict(size=9, color=NO_DATA), text=none["name"],
        hovertemplate="<b>%{text}</b><br>No wait posted<extra></extra>"))
    over = has["wait_min"] >= threshold
    fig.add_trace(go.Scattermap(
        lat=has["lat"], lon=has["lon"], mode="markers", name="Posted wait",
        marker=dict(size=np.where(over, 22, 16), color=has["wait_min"], colorscale=BLUES,
                    cmin=0, cmax=max(240, float(has["wait_min"].max() or 0)),
                    colorbar=dict(title="Wait (min)", thickness=12, len=0.6)),
        text=has["name"], customdata=np.stack([has["wait_min"], has["type"]], axis=1) if len(has) else None,
        hovertemplate="<b>%{text}</b><br>%{customdata[0]:.0f} min wait · %{customdata[1]}<extra></extra>"))
    lat = d["lat"].median() if len(d) else 49.25
    lon = d["lon"].median() if len(d) else -123.1
    fig.update_layout(map=dict(style="carto-positron", center=dict(lat=lat, lon=lon), zoom=9.2),
                      height=440, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=SURFACE,
                      legend=dict(orientation="h", y=0.01, x=0.01, bgcolor="rgba(252,252,251,0.85)"))
    return fig


# ---------- Explore ----------

def heatmap(w: pd.DataFrame, title_site: str) -> go.Figure:
    if w.empty:
        return empty("No data for these filters")
    t = w["observed_at"].dt.tz_convert(PT)
    d = w.assign(day=t.dt.dayofweek, hour=t.dt.hour)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    grid = d.pivot_table(index="day", columns="hour", values="wait_min", aggfunc="mean").reindex(
        index=range(7), columns=range(24))
    peak = d.groupby("hour")["wait_min"].mean()
    title = (f"{title_site}: waits peak around {_hour_label(int(peak.idxmax()))}"
             if len(peak) else title_site)
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[_hour_label(h) for h in range(24)], y=days, colorscale=BLUES,
        colorbar=dict(title="Avg wait (min)", thickness=12), xgap=2, ygap=2,
        hovertemplate="%{y} %{x}<br>Average %{z:.0f} min<extra></extra>"))
    style(fig, title, y_title=None, height=340)
    fig.update_yaxes(autorange="reversed", gridcolor=SURFACE)
    fig.update_layout(hovermode="closest")
    return fig


def timeseries(w: pd.DataFrame, site: str) -> go.Figure:
    if w.empty:
        return empty("No data for these filters")
    if site == "All sites":
        s = w.groupby("observed_at")["wait_min"].mean().reset_index()
        name, title = "Average across reporting sites", "Average posted wait across reporting sites"
    else:
        s = w[["observed_at", "wait_min"]].sort_values("observed_at")
        name, title = site, f"{site}: posted wait over time"
    fig = go.Figure(go.Scatter(x=to_pt(s["observed_at"]), y=s["wait_min"], mode="lines", name=name,
                               line=dict(color=INK, width=2),
                               hovertemplate="%{y:.0f} min<extra></extra>"))
    return style(fig, title, height=340)


def dry_vs_rain(w: pd.DataFrame, wx: pd.DataFrame) -> go.Figure:
    if w.empty or wx.empty:
        return empty("No weather data yet")
    d = w.assign(hour=w["observed_at"].dt.floor("1h")).merge(
        wx.rename(columns={"time": "hour"}), on="hour", how="inner")
    if d.empty:
        return empty("No overlapping weather data yet")
    d["sky"] = np.where(d["precip_mm"] > 0.1, "Raining", "Dry")
    g = d.groupby("sky").agg(wait=("wait_min", "mean"), hours=("hour", "nunique")).reindex(["Dry", "Raining"])
    diff = g.loc["Raining", "wait"] - g.loc["Dry", "wait"] if g["wait"].notna().all() else np.nan
    if np.isnan(diff):
        title = "Not enough rainy hours yet to compare"
    elif abs(diff) < 5:
        title = "Rain makes little difference to waits so far"
    else:
        title = f"Waits run {abs(diff):.0f} min {'longer' if diff > 0 else 'shorter'} when it rains"
    fig = go.Figure(go.Bar(
        x=g.index, y=g["wait"], marker=dict(color=["#86b6ef", "#1c5cab"], cornerradius=4),
        text=[f"{v:.0f} min<br>{int(h) if pd.notna(h) else 0} h of data" for v, h in zip(g["wait"].fillna(0), g["hours"])],
        textposition="outside", hovertemplate="%{x}: %{y:.0f} min<extra></extra>", width=0.5))
    style(fig, title, height=340)
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_yaxes(range=[0, float(np.nanmax(g["wait"].values) * 1.3) if g["wait"].notna().any() else 1])
    return fig


# ---------- Forecast vs actual ----------

def pred_vs_actual(actual: pd.DataFrame, preds: pd.DataFrame, models: list[str], label: str,
                   horizon: int) -> go.Figure:
    if actual.empty:
        return empty("No data for these filters")
    fig = go.Figure(go.Scatter(x=to_pt(actual["t"]), y=actual["actual"], mode="lines", name="Actual wait",
                               line=dict(color=INK, width=2.5),
                               hovertemplate="Actual %{y:.0f} min<extra></extra>"))
    for m in models:
        p = preds[preds["model_name"] == m]
        if p.empty:
            continue
        fig.add_trace(go.Scatter(
            x=to_pt(p["target_time"]), y=p["predicted"], mode="lines", name=MODEL_LABEL[m],
            line=dict(color=MODEL_COLOR[m], width=2, dash=MODEL_DASH[m]),
            hovertemplate=f"{MODEL_LABEL[m]} %{{y:.0f}} min<extra></extra>"))
    now = pd.Timestamp.now(tz=PT).tz_localize(None)
    fig.add_vline(x=now, line=dict(color=INK_2, width=1, dash="dot"))
    fig.add_annotation(x=now, y=1, yref="paper", text="now", showarrow=False, xanchor="left",
                       font=dict(color=INK_2, size=11))
    return style(fig, f"{label}: forecast made {HORIZON_LABEL[horizon]} ahead vs what happened", height=400)


def rolling_mae(s: pd.DataFrame, models: list[str], horizon: int) -> go.Figure:
    s = s[(s["horizon_min"] == horizon) & s["model_name"].isin(models)]
    if s.empty:
        return empty("No forecasts have been scored yet. Check back in a few hours.")
    fig = go.Figure()
    last = {}
    for m in models:
        d = s[s["model_name"] == m]
        if d.empty:
            continue
        h = d.set_index("target_time").resample("1h")["abs_error"].agg(["sum", "count"])
        r = h.rolling(24, min_periods=1).sum()
        mae = (r["sum"] / r["count"]).where(r["count"] >= 5)
        last[m] = mae.dropna().iloc[-1] if mae.notna().any() else np.nan
        fig.add_trace(go.Scatter(x=to_pt(mae.index.to_series()), y=mae, mode="lines", name=MODEL_LABEL[m],
                                 line=dict(color=MODEL_COLOR[m], width=2, dash=MODEL_DASH[m]),
                                 hovertemplate=f"{MODEL_LABEL[m]} %{{y:.1f}} min<extra></extra>"))
    title = "Average error over the last 24 hours (lower is better)"
    if "main" in last and "persistence" in last and pd.notna(last["main"]) and pd.notna(last["persistence"]):
        diff = last["persistence"] - last["main"]
        title = (f"Main model now misses by {last['main']:.0f} min, "
                 f"{abs(diff):.0f} min {'less' if diff >= 0 else 'more'} than assuming no change")
    return style(fig, title, y_title="Mean absolute error (minutes)", height=380)


def correction_bars(s: pd.DataFrame) -> go.Figure:
    m = s[s["model_name"] == "main"]
    if m.empty:
        return empty("No main model forecasts scored yet")
    g = m.groupby("horizon_min").agg(raw=("abs_error_raw", "mean"), corrected=("abs_error", "mean"))
    x = [HORIZON_LABEL[h] for h in g.index]
    fig = go.Figure([
        go.Bar(x=x, y=g["raw"], name="Before bias correction", marker=dict(color="#86b6ef", cornerradius=4),
               text=[f"{v:.1f}" for v in g["raw"]], textposition="outside"),
        go.Bar(x=x, y=g["corrected"], name="After bias correction", marker=dict(color="#1c5cab", cornerradius=4),
               text=[f"{v:.1f}" for v in g["corrected"]], textposition="outside"),
    ])
    change = (g["corrected"] - g["raw"]).mean()
    title = (f"Bias correction {'cuts' if change < 0 else 'adds'} {abs(change):.1f} min of error on average"
             if abs(change) >= 0.1 else "Bias correction makes almost no difference so far")
    style(fig, title, y_title="Mean absolute error (minutes)", height=340)
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, hovermode="closest")
    fig.update_yaxes(range=[0, float(max(g["raw"].max(), g["corrected"].max())) * 1.25])
    fig.update_xaxes(title="Forecast horizon")
    return fig


def error_by_hour(s: pd.DataFrame, horizon: int) -> go.Figure:
    m = s[(s["model_name"] == "main") & (s["horizon_min"] == horizon)]
    if m.empty:
        return empty("No main model forecasts scored yet")
    g = m.assign(hour=m["target_time"].dt.tz_convert(PT).dt.hour).groupby("hour")["abs_error"].mean()
    worst = int(g.idxmax())
    fig = go.Figure(go.Bar(x=[_hour_label(h) for h in g.index], y=g.values,
                           marker=dict(color=MODEL_COLOR["main"], cornerradius=4),
                           hovertemplate="%{x}: %{y:.1f} min<extra></extra>"))
    style(fig, f"The model struggles most around {_hour_label(worst)}", y_title="Mean absolute error (minutes)",
          height=340)
    fig.update_layout(hovermode="closest", bargap=0.15)
    fig.update_xaxes(title="Hour the forecast was for")
    return fig


def _hour_label(h: int) -> str:
    return f"{(h % 12) or 12} {'am' if h < 12 else 'pm'}"
