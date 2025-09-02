import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"  # open in default browser


def plot_coords(lat_lon, style="open-street-map", height=600):
    lats, lons = zip(*lat_lon)
    df = pd.DataFrame({"lat": lats, "lon": lons})

    fig = go.Figure(
        [
            go.Scattermap(lat=df.lat, lon=df.lon, mode="lines", name="Path"),
            go.Scattermap(lat=df.lat, lon=df.lon, mode="markers", name="Points"),
        ]
    )

    west, east = float(df.lon.min()), float(df.lon.max())
    south, north = float(df.lat.min()), float(df.lat.max())
    center = {"lat": (south + north) / 2, "lon": (west + east) / 2}

    # Initial zoom heuristic from span (no clamping)
    span = max(east - west, north - south, 1e-6)
    zoom = max(0, 8 - math.log2(span)) + 2  # tweak "+2" if you want tighter/looser

    fig.update_layout(
        map=dict(style=style, center=center, zoom=zoom),
        height=height,
        margin=dict(t=0, r=0, b=0, l=0),
        hovermode="closest",
    )
    return fig


# Example
# coords = [(37.7749, -122.4194), (34.0522, -118.2437), (36.1699, -115.1398), (40.7128, -74.0060)]
gimu_beh_file = Path("/home/fatemeh/Downloads/bird/data/ssl/gimu_behavior/gull/298.csv")
df = pd.read_csv(gimu_beh_file, header=None)
df = df.sort_values([0, 1, 2]).reset_index(drop=True)
lat_longs = df.iloc[::20, -2:].values
coords = [(i[0], i[1]) for i in lat_longs]
fig = plot_coords(coords)
fig.show()
