# pip install -U plotly dash dash-extensions pandas numpy

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash_extensions import EventListener
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

# group by [0, 1]
groups = df.groupby(by=[0, 1], sort=True)
GROUP_KEYS = list(groups.groups.keys())
N = len(GROUP_KEYS)

# --- precompute sorted groups once (faster playback) ---
PRE = {}
for key in GROUP_KEYS:
    PRE[key] = groups.get_group(key).sort_values([0, 1, 2]).reset_index(drop=True)


# derive a (lat, lon) per group from columns [10, 11]
def latlon_for_group(key):
    g = PRE[key]
    lat, lon = g.iloc[0, [10, 11]]
    return float(lat), float(lon)


COORDS = [latlon_for_group(k) for k in GROUP_KEYS]


# --- plotly version of your labeled IMU plot (right subplot), fast + fixed colors ---
def add_labeled_traces_and_decorations(fig: go.Figure, g: pd.DataFrame, glen: int = 20):
    y_min, y_max = -3.5, 3.5

    data = g[[4, 5, 6]].to_numpy()
    indices = g[2].to_numpy()

    # IMU traces with fixed colors + star markers (match "r-*", "b-*", "g-*")
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=data[:, 0],
            mode="lines+markers",
            name="IMU X",
            line=dict(width=1.5, color="red"),
            marker=dict(size=5, symbol="star", color="red"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=data[:, 1],
            mode="lines+markers",
            name="IMU Y",
            line=dict(width=1.5, color="blue"),
            marker=dict(size=5, symbol="star", color="blue"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=data[:, 2],
            mode="lines+markers",
            name="IMU Z",
            line=dict(width=1.5, color="green"),
            marker=dict(size=5, symbol="star", color="green"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    x0, x1 = int(indices[0]), int(indices[-1]) + 1

    # horizontal zero line (black)
    fig.add_trace(
        go.Scatter(
            x=[x0, indices[-1]],
            y=[0, 0],
            mode="lines",
            line=dict(color="black", width=1),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Build all vertical segments as ONE trace (black)
    vx, vy = [], []
    lbl_x, lbl_text = [], []
    last_e = x0
    for i in range(0, len(g), glen):
        s_ind = int(g.iloc[i, 2])
        e_ind = s_ind + glen
        last_e = e_ind
        vx += [s_ind, s_ind, None]
        vy += [y_min, y_max, None]

        label_id = int(g.iloc[i, 3])
        pred_id = int(g.iloc[i, 8])
        prob = float(g.iloc[i, 9])
        label_txt = ind2name[label_id] if label_id != -1 else None
        pred_txt = ind2name.get(pred_id, str(pred_id))
        disp = (
            f"{label_txt}, {pred_txt}, {prob:.3f}"
            if label_txt
            else f"{pred_txt}, {prob:.3f}"
        )
        lbl_x.append(s_ind + (glen + 1) // 4)
        lbl_text.append(disp)

    # closing vertical line
    vx += [last_e, last_e]
    vy += [y_min, y_max]

    fig.add_trace(
        go.Scatter(
            x=vx,
            y=vy,
            mode="lines",
            line=dict(width=1, color="black"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Labels as ONE text trace (black)
    fig.add_trace(
        go.Scatter(
            x=lbl_x,
            y=[y_max - 0.5] * len(lbl_x),
            mode="text",
            text=lbl_text,
            textposition="top center",
            textfont=dict(color="black", size=10),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    # Axes styling
    fig.update_xaxes(range=[x0, x1], dtick=glen, row=1, col=2)
    fig.update_yaxes(
        range=[y_min, y_max], tickmode="array", tickvals=[y_min, 0, y_max], row=1, col=2
    )

    gps_val = float(g.iloc[0, 7]) if 7 in g.columns else float("nan")
    subtitle = (
        f"{g.iloc[0,0]}, {g.iloc[0,1]}, gps:{gps_val:.2f}"
        if not np.isnan(gps_val)
        else f"{g.iloc[0,0]}, {g.iloc[0,1]}"
    )
    return subtitle


# --- make the full 1×2 figure for a given group index ---
def make_figure(idx: int, zoom: int = 15):
    key = GROUP_KEYS[idx % N]
    g = PRE[key]
    lat, lon = COORDS[idx % N]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "map"}, {"type": "xy"}]],
        column_widths=[0.6, 0.4],
        horizontal_spacing=0.06,
    )

    # (A) Map with fixed red cross
    fig.add_trace(
        go.Scattermap(
            lat=[lat],
            lon=[lon],
            mode="markers",
            marker=dict(symbol="cross", size=26, color="red"),
            name="Location",
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        map=dict(style="open-street-map", center=dict(lat=lat, lon=lon), zoom=zoom),
        height=600,
        margin=dict(l=40, r=40, t=40, b=30),
        showlegend=False,
        uirevision="keep",  # keep UI state between updates
        transition={"duration": 0},  # no animation -> snappier
    )

    # (B) Right subplot (your labeled chart)
    subtitle = add_labeled_traces_and_decorations(fig, g, glen=20)

    # lock the right subplot so scroll goes to the map
    fig.update_xaxes(fixedrange=True, row=1, col=2)
    fig.update_yaxes(fixedrange=True, row=1, col=2)

    # subtitle above the RIGHT subplot
    xaxis = getattr(fig.layout, "xaxis2", None) or fig.layout.xaxis
    x0, x1 = xaxis.domain
    fig.add_annotation(
        text=subtitle,
        x=(x0 + x1) / 2,
        xref="paper",
        y=1.02,
        yref="paper",
        showarrow=False,
        font=dict(size=12, color="black"),
    )

    return fig, (lat, lon), key


# Listen for global key presses (Left/Right/Space/Enter)
key_events = [{"event": "keydown", "props": ["key"]}]

app = Dash(__name__)
app.layout = html.Div(
    [
        EventListener(id="keys", events=key_events),  # global key listener
        dcc.Store(id="idx", data=0),  # current group index
        dcc.Store(id="playing", data=False),  # play/pause state
        html.Div(
            [
                html.Button(
                    "⬅ Prev", id="prev", n_clicks=0, style={"marginRight": "8px"}
                ),
                html.Button(
                    "▶ Play", id="play", n_clicks=0, style={"marginRight": "8px"}
                ),
                html.Button("Next ➡", id="next", n_clicks=0),
                dcc.Input(
                    id="jump",
                    type="number",
                    min=0,
                    max=N - 1,
                    step=1,
                    value=0,
                    debounce=True,
                    placeholder=f"0..{N-1}",
                    style={"width": "90px", "marginLeft": "12px"},
                ),
                html.Span(f"/ {N-1}", style={"marginLeft": "6px"}),
                html.Span(id="label", style={"marginLeft": "12px"}),
            ],
            style={"marginBottom": "8px"},
        ),
        dcc.Interval(id="ticker", interval=500, disabled=True),  # 500 ms
        dcc.Graph(id="fig", figure=make_figure(0)[0], config={"scrollZoom": True}),
    ]
)


def _normalize_jump(val, idx, N):
    try:
        v = int(val)
    except (TypeError, ValueError):
        return idx
    return max(0, min(N - 1, v))  # clamp to 0..N-1


# Keep input synced AFTER idx changes (won't fight with typing)
@app.callback(
    Output("jump", "value", allow_duplicate=True),
    Input("idx", "data"),
    prevent_initial_call=True,
)
def sync_jump(idx):
    return int(idx)


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
def step(
    prev_clicks,
    next_clicks,
    play_clicks,
    _tick,
    _n_events,
    jump_val,
    key_event,
    idx,
    playing,
):
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
            idx = _normalize_jump(jump_val, idx, N)  # jump on Enter
        else:
            # other keys (digits, etc.) -> no redraw so typing isn't interrupted
            return (no_update, no_update, no_update, no_update, no_update, no_update)

    elif trig == "ticker":
        if playing:
            idx = (idx + 1) % N
        else:
            return (no_update, no_update, no_update, no_update, no_update, no_update)

    elif trig == "jump":
        idx = _normalize_jump(jump_val, idx, N)

    fig, (lat, lon), key = make_figure(idx)
    label = f"idx={idx} (0..{N-1}) — key={key} — lat,lon={lat:.6f},{lon:.6f}"
    play_text = "⏸ Pause" if playing else "▶ Play"
    ticker_disabled = not playing

    return idx, fig, label, playing, play_text, ticker_disabled


if __name__ == "__main__":
    app.run(debug=True)
