"behavior_classes.png"

import matplotlib.pyplot as plt
import pandas as pd

from behavior import utils as bu

df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/starts.csv", header=None)
for i in [0, 1, 2, 3, 4, 5, 6, 8, 9]:
    cut = df[df[3] == i].iloc[200:220]
    bu.plot_one(cut.iloc[:, 4:].values)
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(
        f"class:{bu.ind2name[cut.iloc[0,3]]}, gps speed:{round(cut.iloc[0,7],2)} m/s"
    )
    fig.set_size_inches([4.94, 2.55])
    plt.show(block=False)

# sorted(df[df[3]==0].iloc[::20,7].values.tolist()) # gps speed checks
# visualize all behaviors separately not gt2
