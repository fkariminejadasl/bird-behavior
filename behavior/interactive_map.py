# pip install -U plotly dash
# pip install anywidget ipywidgets

from dash import Dash, dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- your coordinate list (lat, lon) ---
COORDS = [
    (52.36977, 4.9954),   # Amsterdam-ish
    (52.00947, 4.34438),  # Delft-ish
    (52.37022, 4.89517),  # Dam Square
]

def make_figure(idx: int, zoom: int = 15):
    lat, lon = COORDS[idx % len(COORDS)]
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "map"}, {"type": "xy"}]],
        column_widths=[0.6, 0.4],
        horizontal_spacing=0.06,
    )

    # (A) Map with a red cross at current coord (MapLibre)
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
        map=dict(
            style="open-street-map",
            center=dict(lat=lat, lon=lon),
            zoom=zoom,
        ),
        height=560,
        margin=dict(l=40, r=40, t=30, b=30),
        showlegend=False,
    )

    # (B) Your other figure (dummy)
    fig.add_trace(
        go.Scatter(y=[0, 1, 0, 1], mode="lines+markers", name="Dummy"),
        row=1, col=2
    )
    # Make the right pane ignore scroll zoom so the map captures the wheel
    fig.update_xaxes(fixedrange=True, row=1, col=2)
    fig.update_yaxes(fixedrange=True, row=1, col=2)
    return fig

app = Dash(__name__)
app.layout = html.Div([
    dcc.Store(id="idx", data=0),
    html.Div([
        html.Button("⬅ Prev", id="prev", n_clicks=0, style={"marginRight": "8px"}),
        html.Button("Next ➡", id="next", n_clicks=0),
        html.Span(id="label", style={"marginLeft": "12px"})
    ], style={"marginBottom": "8px"}),
    dcc.Graph(id="fig", figure=make_figure(0), config={"scrollZoom": True}),
])

@app.callback(
    Output("idx", "data"),
    Output("fig", "figure"),
    Output("label", "children"),
    Input("prev", "n_clicks"),
    Input("next", "n_clicks"),
    State("idx", "data")
)
def step(prev_clicks, next_clicks, idx):
    # Which button was pressed?
    trig = ctx.triggered_id
    if trig == "prev":
        idx = (idx - 1) % len(COORDS)
    elif trig == "next":
        idx = (idx + 1) % len(COORDS)
    # Build updated figure centered on the new coordinate
    fig = make_figure(idx)
    lat, lon = COORDS[idx]
    label = f"Point {idx+1}/{len(COORDS)} — lat={lat:.5f}, lon={lon:.5f}"
    return idx, fig, label

if __name__ == "__main__":
    app.run(debug=True)
