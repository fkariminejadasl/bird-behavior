"""Check the Python IMU calibration against the R output, on Ornitela data.

`behavior/data.py::acc_calibrate` is a translation of the R functions of Eldar
Rakhimberdiev (calibration_functions.R). R is not needed here: the R output
already exists. `calibration_per_device_auto.csv` holds the parameters R found
per device, and `all_devices_calibrated.csv` keeps the raw `acc_x, acc_y, acc_z`
they were found from.

Per device, it prints the Python scale next to the R scale, and the scale with
q_static 0.10 (the R default) next to 0.85 (the R example, and what the R output
used). Only the scale is compared: on these on-bird tags R fell back to one
scale and no offset for every device.
"""

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from behavior import data as bd


def main(cfg):
    ref = pd.read_csv(cfg.ref_file).set_index("File")
    df = pd.read_csv(
        cfg.data_file,
        usecols=["device_id", "acc_x", "acc_y", "acc_z"],
        engine="pyarrow",
    )

    print("device  s_R       s_py      diff     s(q=0.10)  q diff%  coverage  method")
    diffs = []
    for device_id, d in df.groupby("device_id", sort=False):
        s_ref = ref.loc[device_id, "sx"]
        res = bd.acc_calibrate(d.acc_x, d.acc_y, d.acc_z, fs=cfg.fs, q_static=0.85)
        res10 = bd.acc_calibrate(d.acc_x, d.acc_y, d.acc_z, fs=cfg.fs, q_static=0.10)
        p, s10 = res["params"], res10["params"]["sx"]
        diffs.append(abs(p["sx"] - s_ref))
        print(
            f"{device_id}  {s_ref:8.2f}  {p['sx']:8.2f}  {p['sx'] - s_ref:7.1e}"
            f"  {s10:8.2f}  {100 * (s10 - p['sx']) / p['sx']:6.2f}"
            f"  {res['diagnostics']['orientation_coverage']:8.4f}  {p['method']}"
        )
    print(f"max |s_py - s_R| over {len(diffs)} devices: {np.max(diffs):.1e}")


if __name__ == "__main__":
    config = {
        "data_file": "/home/fatemeh/Downloads/bird/data/simon/all_devices_calibrated.csv",
        "ref_file": "/home/fatemeh/Downloads/bird/data/simon/calibration_per_device_auto.csv",
        "fs": 20,  # Hz
    }
    main(OmegaConf.create(config))
