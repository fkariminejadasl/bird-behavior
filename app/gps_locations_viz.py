# pip install dash plotly pandas

import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html  # ensure html is imported

# ---------- Load full data ----------
gimu_beh_file = Path(
    "/home/fatemeh/Downloads/bird/data/ssl/gimu_behavior/gull/6210_72.csv"
)
df_all = (
    pd.read_csv(gimu_beh_file, header=None)
    .sort_values([0, 1, 2])
    .reset_index(drop=True)
)
df_all[1] = pd.to_datetime(df_all[1])  # timestamps
df_all["lat"] = df_all[10]  # column 10 = latitude
df_all["lon"] = df_all[11]  # column 11 = longitude
df_all["val"] = df_all[8].astype(int)  # column 8 = behavior class

# ---------- Initial map view (from ALL data; free pan/zoom) ----------
west, east = float(df_all.lon.min()), float(df_all.lon.max())
south, north = float(df_all.lat.min()), float(df_all.lat.max())
center = {"lat": (south + north) / 2.0, "lon": (west + east) / 2.0}
span = max(east - west, north - south, 1e-6)
zoom = max(0, 8 - math.log2(span)) + 2

# ---------- Plotting dataframe (optional downsample) ----------
df = df_all.iloc[::20].reset_index(drop=True)

# ---------- Class maps (curated) ----------
# fmt: off
# # colors from tab20
# import matplotlib.colors as mcolors
# import matplotlib.pyplot as plt
# tab20 = plt.get_cmap("tab20")
# ind2color = {i: mcolors.to_hex(tab20(i)) for i in range(10)}
ind2color = {
    0:'#1f77b4', 1:'#aec7e8', 2:'#ff7f0e', 3:'#ffbb78', 4:'#2ca02c',
    5:'#98df8a', 6:'#d62728', 8: '#9467bd', 9: '#c5b0d5'
}
ind2name = {
    0: "Flap", 1: "ExFlap", 2: "Soar", 3: "Boat", 4: "Float",
    5: "SitStand", 6: "TerLoco", 8: "Manouvre", 9: "Pecking"
}
# fmt: on
class_keys = sorted(ind2name.keys())

# ---------- Time slider bounds ----------
# derive min/max directly as UTC timestamps
t_min = int(df_all[1].min().replace(tzinfo=timezone.utc).timestamp())
t_max = int(df_all[1].max().replace(tzinfo=timezone.utc).timestamp())


# ---------- Figure helper ----------
def make_figure(dff: pd.DataFrame, style="open-street-map", height=650) -> go.Figure:
    traces = [
        go.Scattermap(
            lat=dff["lat"],
            lon=dff["lon"],
            mode="lines",
            name="Path",
            line=dict(width=1, color="rgba(0,0,0,0.35)"),
            hoverinfo="skip",
        )
    ]
    for k in class_keys:
        sub = dff[dff["val"] == k]
        lat_vals = sub["lat"] if len(sub) else [None]
        lon_vals = sub["lon"] if len(sub) else [None]
        traces.append(
            go.Scattermap(
                lat=lat_vals,
                lon=lon_vals,
                mode="markers",
                name=ind2name[k],
                marker=dict(size=6, color=ind2color[k]),
                showlegend=True,
            )
        )

    fig = go.Figure(traces)
    fig.update_layout(
        map=dict(style=style, center=center, zoom=zoom),
        height=height,
        margin=dict(t=0, r=0, b=0, l=0),
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        uirevision="keep",  # keep user view on updates
    )
    return fig


# ---------- Dash app ----------
app = Dash(__name__)

# Hide the slider's numeric handle tooltip bubble
app.index_string = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>.rc-slider-tooltip{display:none !important;}</style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>"""

# --- compact slider labels (strings only; no HTML) ---
LABEL_MODE = "day"  # or "month"


def mk_marks(ts: pd.Series, max_marks=12):
    ts_full = ts.sort_values().dt.floor("S").drop_duplicates()

    if LABEL_MODE == "month":
        ts_norm = ts_full.dt.to_period("M").dt.start_time.drop_duplicates()
        fmt = "%Y-%m"
    else:
        ts_norm = ts_full.dt.normalize().drop_duplicates()
        fmt = "%Y-%m-%d"

    if len(ts_norm) > max_marks:
        step = max(1, len(ts_norm) // (max_marks - 1))
        ts_norm = ts_norm[::step]

    endpoints = pd.Series([ts_full.iloc[0], ts_full.iloc[-1]])
    ts_marks = (
        pd.concat([pd.Series(ts_norm), endpoints]).drop_duplicates().sort_values()
    )

    # convert to seconds since epoch in UTC
    return {
        int(t.replace(tzinfo=timezone.utc).timestamp()): t.strftime(fmt)
        for t in ts_marks
    }


app.layout = html.Div(
    style={"padding": "12px", "fontFamily": "system-ui, sans-serif"},
    children=[
        dcc.Graph(id="map-graph", figure=make_figure(df)),
        html.Div("Time window", style={"padding": "8px 4px 0 4px"}),
        dcc.RangeSlider(
            id="time-slider",
            min=t_min,
            max=t_max,
            step=1,
            allowCross=False,
            value=[t_min, t_max],
            marks=mk_marks(df_all[1]),
            tooltip={"always_visible": False, "placement": "bottom"},
        ),
        html.Div(id="time-readout", style={"marginTop": "6px", "opacity": 0.7}),
    ],
)


@app.callback(
    Output("map-graph", "figure"),
    Output("time-readout", "children"),
    Input("time-slider", "value"),
)
def update_map(selected_range):
    start_s, end_s = selected_range
    start = datetime.fromtimestamp(start_s, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    end = datetime.fromtimestamp(end_s, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    mask = (df[1] >= start) & (df[1] <= end)
    dff = df.loc[mask].copy()
    fig = make_figure(dff)
    txt = f"Showing: {start} → {end}  (rows: {len(dff)})"
    return fig, txt


if __name__ == "__main__":
    app.run(debug=True)
