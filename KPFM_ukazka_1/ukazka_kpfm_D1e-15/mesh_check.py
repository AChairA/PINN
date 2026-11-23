"""Forward-grid convergence at fixed sites/times, independent of neural training."""
import json
from pathlib import Path
import numpy as np
from physics import Surface
from design import read_csv, write_csv

ROOT = Path(__file__).resolve().parent


def main():
    out = ROOT/"vystupy"
    c = json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
    rows = read_csv(out/"measurement_plan.csv")
    points = np.array([[float(r["x_um"]), float(r["y_um"])] for r in rows])
    times = np.array(c["design"]["nominal_times_s"])
    D = c["diagnostics"]["D_nominal_m2_s"]
    cases = [(20, .025), (40, .025), (40, .0125), (80, .0125)]
    records, predictions = [], []
    for base, minimum in cases:
        m = dict(c["model"], edge_min_step_um=minimum)
        surface = Surface(m, base)
        print(f"Mesh base={base}, edge={minimum} um, actual={surface.n}x{surface.n}", flush=True)
        fields = surface.simulate(times, D)
        signal = c["model"]["gain_initial"]*(surface.psf(points) @ fields.T)
        predictions.append(signal)
        records.append({"base_grid": base, "edge_min_step_um": minimum,
                        "actual_grid": surface.n, "dielectric_cells": int(surface.keep.sum())})
    finest = predictions[-1]
    for record, prediction in zip(records, predictions):
        error = prediction-finest
        record.update(RMSE_to_finest_V=float(np.sqrt(np.mean(error**2))),
                      max_abs_error_to_finest_V=float(abs(error).max()))
    write_csv(out/"mesh_convergence.csv", records)
    (out/"mesh_convergence.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2), flush=True)


if __name__ == "__main__":
    main()
