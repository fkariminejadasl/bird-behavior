# pip install -U plotly

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ensure interactive output; pick one that fits your environment
# pio.renderers.default = "notebook_connected"  # in Jupyter
# pio.renderers.default = "vscode"              # in VS Code
# pio.renderers.default = "browser"             # from a script

lat, lon = 52.36977, 4.9954
zoom = 15

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "map"}, {"type": "xy"}]],
    column_widths=[0.6, 0.4],
    horizontal_spacing=0.06,
)

# (A) Interactive map with a red cross
fig.add_trace(
    go.Scattermap(
        lat=[lat], lon=[lon],
        mode="markers",
        marker=dict(symbol="cross", size=24, color="red"),
        name="Location",
    ),
    row=1, col=1
)

# (B) Your other figure (dummy)
fig.add_trace(
    go.Scatter(y=[0, 1, 0, 1], mode="lines+markers", name="Dummy"),
    row=1, col=2
)

# Important bits:
# - give the map a tile style (MapLibre)
# - disable zoom/pan on the cartesian subplot so the scroll wheel goes to the map
fig.update_layout(
    map=dict(
        style="open-street-map",      # tile source (no token needed)
        center=dict(lat=lat, lon=lon),
        zoom=zoom,
    ),
    height=550,
    margin=dict(l=40, r=40, t=40, b=40),
    showlegend=False,
)

# prevent the right subplot from zooming/panning
fig.update_xaxes(fixedrange=True, row=1, col=2)
fig.update_yaxes(fixedrange=True, row=1, col=2)

# enable scroll zoom just in case your environment needs it
fig.show(config={"scrollZoom": True})
