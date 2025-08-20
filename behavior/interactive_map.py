# pip install -U plotly dash dash-extensions pandas numpy

from pathlib import Path
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
from dash_extensions import EventListener
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- labels (your mapping) ---
# fmt: off
ind2name = {
    0: "Flap", 1: "ExFlap", 2: "Soar", 3: "Boat", 4: "Float",
    5: "SitStand", 6: "TerLoco", 7: "Other", 8: "Manouvre", 9: "Pecking",
}
# fmt: on

# --- load your CSV ---
# gimu_beh_file = Path("/home/fatemeh/Downloads/bird/data/final/proc2/starts_gimu_behavior.csv")
gimu_beh_file = Path("/home/fatemeh/Downloads/bird/data/ssl/gimu_behavior/gull/298.csv")
df = pd.read_csv(gimu_beh_file, header=None)

# group by [0, 1] like your code
groups = df.groupby(by=[0, 1], sort=True)
GROUP_KEYS = list(groups.groups.keys())
N = len(GROUP_KEYS)

# derive a (lat, lon) per group from columns [10, 11]
def latlon_for_group(key):
    g = groups.get_group(key)
    lat, lon = g.iloc[0, [10, 11]]  # assumes these columns exist
    return float(lat), float(lon)

COORDS = [latlon_for_group(k) for k in GROUP_KEYS]

# --- plotly version of your labeled IMU plot (right subplot) ---
def add_labeled_traces_and_decorations(fig: go.Figure, g: pd.DataFrame, glen: int = 20):
    y_min, y_max = -3.5, 3.5

    g = g.sort_values([0, 1, 2]).reset_index(drop=True)
    data = g[[4, 5, 6]].to_numpy()
    indices = g[2].to_numpy()

    # IMU traces
    for y, name, color in zip(
        data.T, ["IMU X", "IMU Y", "IMU Z"], ["red", "blue", "green"]
    ):
        fig.add_trace(
            go.Scatter(
                x=indices, y=y,
                mode="lines+markers",
                name=name,
                line=dict(width=1.5, color=color),
                marker=dict(size=5),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1, col=2
        )

    x0, x1 = int(indices[0]), int(indices[-1]) + 1

    # horizontal zero line
    fig.add_trace(
        go.Scatter(
            x=[x0, indices[-1]], y=[0, 0],
            mode="lines",
            line=dict(color="black", width=1),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=2
    )

    # vertical lines + text per segment
    last_e = x0
    for i in range(0, len(g), glen):
        s_ind = int(g.iloc[i, 2])
        e_ind = s_ind + glen
        last_e = e_ind

        label_id = int(g.iloc[i, 3])
        pred_id  = int(g.iloc[i, 8])
        prob     = float(g.iloc[i, 9])

        label_txt = ind2name[label_id] if label_id != -1 else None
        pred_txt  = ind2name.get(pred_id, str(pred_id))
        disp = f"{label_txt}, {pred_txt}, {prob:.3f}" if label_txt else f"{pred_txt}, {prob:.3f}"
        x_text = s_ind + (glen + 1) // 4

        # vertical line
        fig.add_trace(
            go.Scatter(
                x=[s_ind, s_ind], y=[y_min, y_max],
                mode="lines",
                line=dict(color="black", width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1, col=2
        )
        # label text
        fig.add_trace(
            go.Scatter(
                x=[x_text], y=[y_max - 0.5],
                mode="text",
                text=[disp],
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1, col=2
        )

    # closing vertical line
    fig.add_trace(
        go.Scatter(
            x=[last_e, last_e], y=[y_min, y_max],
            mode="lines",
            line=dict(color="black", width=1),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=2
    )

    # axes styling
    fig.update_xaxes(range=[x0, x1], dtick=glen, row=1, col=2)
    fig.update_yaxes(
        range=[y_min, y_max],
        tickmode="array",
        tickvals=[y_min, 0, y_max],
        row=1, col=2
    )

    gps_val = float(g.iloc[0, 7]) if 7 in g.columns else np.nan
    subtitle = (
        f"{g.iloc[0,0]}, {g.iloc[0,1]}, gps:{gps_val:.2f}"
        if not np.isnan(gps_val) else f"{g.iloc[0,0]}, {g.iloc[0,1]}"
    )
    return subtitle

# --- make the full 1×2 figure for a given group index ---
def make_figure(idx: int, zoom: int = 15):
    key = GROUP_KEYS[idx % len(GROUP_KEYS)]
    g = groups.get_group(key)
    lat, lon = COORDS[idx % len(COORDS)]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "map"}, {"type": "xy"}]],
        column_widths=[0.6, 0.4],
        horizontal_spacing=0.06,
    )

    # (A) Map
    fig.add_trace(
        go.Scattermap(
            lat=[lat], lon=[lon],
            mode="markers",
            marker=dict(symbol="cross", size=26, color="red"),
            name="Location",
        ),
        row=1, col=1
    )
    fig.update_layout(
        map=dict(style="open-street-map", center=dict(lat=lat, lon=lon), zoom=zoom),
        height=600, margin=dict(l=40, r=40, t=40, b=30), showlegend=False,
    )

    # (B) Right subplot (your labeled chart)
    subtitle = add_labeled_traces_and_decorations(fig, g, glen=20)

    # lock the right subplot so scroll goes to the map
    fig.update_xaxes(fixedrange=True, row=1, col=2)
    fig.update_yaxes(fixedrange=True, row=1, col=2)

    # --- add subtitle above the RIGHT subplot ---
    # whichever xaxis is the cartesian pane (xaxis or xaxis2), get its domain
    xaxis = getattr(fig.layout, "xaxis2", None) or fig.layout.xaxis
    x0, x1 = xaxis.domain
    fig.add_annotation(
        text=subtitle,
        x=(x0 + x1) / 2,  # center of the right subplot
        xref="paper",
        y=1.02,           # a bit above the top
        yref="paper",
        showarrow=False,
        font=dict(size=12),
    )

    return fig, (lat, lon), key

# Listen for global key presses (Left/Right/Space)
key_events = [{"event": "keydown", "props": ["key"]}]

app = Dash(__name__)
app.layout = html.Div([
    EventListener(id="keys", events=key_events),     # global key listener
    dcc.Store(id="idx", data=0),                     # current group index
    dcc.Store(id="playing", data=False),             # play/pause state

    html.Div([
        html.Button("⬅ Prev", id="prev", n_clicks=0, style={"marginRight": "8px"}),
        html.Button("▶ Play", id="play", n_clicks=0, style={"marginRight": "8px"}),
        html.Button("Next ➡", id="next", n_clicks=0),
        dcc.Input(
            id="jump", type="number",
            min=0, max=N-1, step=1,
            value=0,                   # <- start at 0
            debounce=True, placeholder=f"0..{N-1}",
            style={"width": "90px", "marginLeft": "12px"}
        ),
        html.Span(f"/ {N-1}", style={"marginLeft": "6px"}),
        html.Span(id="label", style={"marginLeft": "12px"}),
    ], style={"marginBottom": "8px"}),

    # interval drives auto-advance while playing
    dcc.Interval(id="ticker", interval=500, disabled=True),  # 500 ms; tweak to taste

    dcc.Graph(id="fig", figure=make_figure(0)[0], config={"scrollZoom": True}),
])


def _normalize_jump(val, idx, N):
    try:
        v = int(val)
    except (TypeError, ValueError):
        return idx
    # clamp to 0..N-1
    return max(0, min(N - 1, v))


@app.callback(
    Output("jump", "value", allow_duplicate=True),
    Input("idx", "data"),
    prevent_initial_call=True,
)
def sync_jump(idx):
    return int(idx)  # <- zero-based display


@app.callback(
    Output("idx", "data"),
    Output("fig", "figure"),
    Output("label", "children"),
    Output("playing", "data"),
    Output("play", "children"),
    Output("ticker", "disabled"),
    Input("prev", "n_clicks"),
    Input("next", "n_clicks"),
    Input("play", "n_clicks"),
    Input("ticker", "n_intervals"),
    Input("keys", "n_events"),
    Input("jump", "value"),
    State("keys", "event"),
    State("idx", "data"),
    State("playing", "data"),
)
def step(prev_clicks, next_clicks, play_clicks, _tick, _n_events, jump_val,
         key_event, idx, playing):
    N = len(GROUP_KEYS)
    trig = ctx.triggered_id

    if trig == "prev":
        idx = (idx - 1) % N

    elif trig == "next":
        idx = (idx + 1) % N

    elif trig == "play":
        playing = not playing

    elif trig == "keys" and isinstance(key_event, dict):
        k = key_event.get("key")
        if k == "ArrowLeft":
            idx = (idx - 1) % N
        elif k == "ArrowRight":
            idx = (idx + 1) % N
        elif k in (" ", "Space", "Spacebar"):
            playing = not playing
        elif k in ("Enter", "NumpadEnter"):
            # Treat Enter like a manual jump to whatever is in the box
            idx = _normalize_jump(jump_val, idx, N)
        else:
            # Any other key (digits, etc.) → do nothing so typing isn't interrupted
            return (no_update, no_update, no_update,
                    no_update, no_update, no_update)

    elif trig == "ticker":
        if playing:
            idx = (idx + 1) % N
        else:
            return (no_update, no_update, no_update,
                    no_update, no_update, no_update)

    elif trig == "jump":
        # fires on Enter/blur because debounce=True
        idx = _normalize_jump(jump_val, idx, N)

    fig, (lat, lon), key = make_figure(idx)
    label = f"idx={idx} (0..{N-1}) — key={key} — lat,lon={lat:.6f},{lon:.6f}"
    play_text = "⏸ Pause" if playing else "▶ Play"
    ticker_disabled = not playing

    return idx, fig, label, playing, play_text, ticker_disabled


if __name__ == "__main__":
    app.run(debug=True)   # Dash ≥ 2.14
