import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"  # open in default browser


def plot_coords(lat_lon, style="open-street-map", height=600):
    lats = [p[0] for p in lat_lon]
    lons = [p[1] for p in lat_lon]
    vals = [p[2] for p in lat_lon]  # df[8] values
    df = pd.DataFrame({"lat": lats, "lon": lons, "val": vals})

    # # colors from tab20 except from 8 and 9 get the values of 7 and 8
    # import matplotlib.colors as mcolors
    # import matplotlib.pyplot as plt
    # tab20 = plt.get_cmap("tab20")
    # ind2color = {i: mcolors.to_hex(tab20(i)) for i in range(10)}
    # fmt: off
    ind2color = {
        0: '#1f77b4', 1: '#aec7e8', 2: '#ff7f0e', 3: '#ffbb78', 4: '#2ca02c',
        5: '#98df8a', 6: '#d62728', 8: '#ff9896', 9: '#9467bd'
    }
    ind2name = {0:"Flap",1:"ExFlap",2:"Soar",3:"Boat",4:"Float",5:"SitStand",6:"TerLoco",8:"Manouvre",9:"Pecking"}
    # fmt: on

    traces = [
        go.Scattermap(  # keep the path as a single (subtle) line
            lat=df.lat,
            lon=df.lon,
            mode="lines",
            name="Path",
            line=dict(width=1, color="rgba(0,0,0,0.35)"),
            hoverinfo="skip",
        )
    ]
    for k in sorted(ind2name.keys()):
        m = df.val == k
        traces.append(
            go.Scattermap(
                lat=df.lat[m],
                lon=df.lon[m],
                mode="markers",
                name=ind2name[k],
                marker=dict(size=6, color=ind2color[k]),
            )
        )

    fig = go.Figure(traces)

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
lat_lon_val = df.iloc[::20, [-2, -1, 8]].values
coords = [(r[0], r[1], r[2]) for r in lat_lon_val]
fig = plot_coords(coords)
fig.show()
