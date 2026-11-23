"""Run: ../.venv/Scripts/python.exe run_demo.py all (from this folder)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from physics import Surface, SpatialBasis
from design import choose_sites, make_measurements, read_csv, write_csv

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "plan", "simulate", "fit", "validate"), nargs="?", default="all")
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    parser.add_argument("--output", type=Path, default=ROOT/"vystupy")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    c = json.loads(args.config.read_text(encoding="utf-8"))
    if args.steps is not None:
        c["training"]["steps"] = args.steps
    validate_config(c)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model, design, synthetic = c["model"], c["design"], c["synthetic_only"]
    save_json(out/"run_config.json", c)
    basis = None
    if args.stage in ("all", "plan", "fit", "validate"):
        print("Building inverse geometry and spatial eigenbasis...", flush=True)
        basis = SpatialBasis(Surface(model, model["inverse_grid"]))
    if args.stage in ("all", "plan"):
        print("1. Designing sites using PSF-aware sensitivity and prior D range...", flush=True)
        sites, candidates, sensitivity = choose_sites(basis, model, design)
        write_csv(out/"measurement_plan.csv", sites)
        np.savez_compressed(out/"design_sensitivity.npz", xy_um=candidates, rms_dV_dlogD_V=sensitivity)
        from plots import plan_plot
        plan_plot(out, sites, candidates, sensitivity, c)
        save_json(out/"acquisition_plan.json", {"nominal_times_s": design["nominal_times_s"],
                  "point_dwell_s": design["point_dwell_s"], "n_sites": len(sites),
                  "scan_duration_s": len(sites)*design["point_dwell_s"],
                  "holdout_nominal_time_s": design["holdout_nominal_time_s"],
                  "PSF_sigma_um": model["psf_sigma_um"], "PSF_FWHM_um": 2.354820045*model["psf_sigma_um"]})
    if args.stage in ("all", "simulate"):
        print("2. Simulating independent fine-grid data, with convolution and noise...", flush=True)
        sites = read_csv(out/"measurement_plan.csv")
        for site in sites:
            site["x_um"], site["y_um"] = float(site["x_um"]), float(site["y_um"])
        truth_model = dict(model)
        truth_model["edge_min_step_um"] = synthetic["truth_edge_min_step_um"]
        surface = Surface(truth_model, synthetic["truth_grid"])
        rows, truth, fields = make_measurements(surface, sites, model, design, synthetic)
        write_csv(out/"measurements.csv", rows)
        write_csv(out/"synthetic_truth_DO_NOT_FIT.csv", truth)
        np.savez_compressed(out/"synthetic_fields_DO_NOT_FIT.npz", potential_V=fields,
                            x_um=surface.axis, y_um=surface.axis, faces_um=surface.faces,
                            times_s=design["nominal_times_s"])
        from plots import data_plot
        data_plot(out, surface, fields, rows, design["nominal_times_s"])
    if args.stage in ("all", "fit"):
        print("3. Training the inverse PINN (no access to synthetic truth)...", flush=True)
        import torch
        from pinn import fit
        rows = read_csv(out/"measurements.csv")
        network, report, history, predictions = fit(basis, rows, model, c["training"])
        save_json(out/"fit_summary.json", report)
        write_csv(out/"loss.csv", history)
        write_csv(out/"predictions.csv", [{**row, "PINN_KPFM_V": float(value)} for row, value in zip(rows, predictions)])
        torch.save({"state_dict": network.state_dict(), "model_config": model,
                    "width": c["training"]["width"], "spatial_grid": model["inverse_grid"],
                    "method": report["method"]}, out/"model.pt")
        # Truth is read only after fit(), for evaluation and visualization.
        truth_path = out/"synthetic_fields_DO_NOT_FIT.npz"
        if truth_path.exists():
            truth_data = np.load(truth_path)
            fields = truth_data["potential_V"]
            comparison = {"D_true_m2_s": synthetic["D_true_m2_s"],
                          "D_PINN_m2_s": report["D_eff_m2_s"],
                          "relative_error": abs(report["D_eff_m2_s"]/synthetic["D_true_m2_s"]-1),
                          "fine_grid": synthetic["truth_grid"], "inverse_grid": model["inverse_grid"]}
            comparison["recovery_within_10_percent"] = comparison["relative_error"] < .1
            comparison["inverse_actual_cells_per_axis"] = basis.surface.n
            comparison["truth_actual_cells_per_axis"] = len(truth_data["x_um"])
            comparison["inverse_min_cell_um"] = float(np.diff(basis.surface.faces).min())
            comparison["truth_min_cell_um"] = float(np.diff(truth_data["faces_um"]).min())
            save_json(out/"synthetic_recovery_check.json", comparison)
            from plots import fit_plots
            fit_plots(out, basis, network, rows, predictions, history, fields, c, report,
                      truth_axis=truth_data["x_um"], truth_faces=truth_data["faces_um"])
        print(json.dumps(report, indent=2), flush=True)
    if args.stage == "validate":
        from pinn import fit
        rows = read_csv(out/"measurements.csv")
        results = []
        for seed, initial in c["training"]["validation_restarts"]:
            print(f"Checking seed={seed}, initial D={initial:g}...", flush=True)
            _, report, _, _ = fit(basis, rows, model, c["training"], seed=seed, D_initial=initial)
            results.append(report)
        print("Control: deliberately omitting spatial convolution...", flush=True)
        _, no_blur, _, _ = fit(basis, rows, model, c["training"], sigma=0.)
        save_json(out/"validation_runs.json", {"PSF_aware_restarts": results, "incorrect_no_blur_fit": no_blur,
                  "seed_spread_is_not_a_confidence_interval": True})
        print(json.dumps({"restarts_D": [r["D_eff_m2_s"] for r in results],
                          "no_blur_D": no_blur["D_eff_m2_s"]}, indent=2), flush=True)
    print(f"Done: {out}", flush=True)


def validate_config(c):
    m, d, s, training = c["model"], c["design"], c["synthetic_only"], c["training"]
    for key in ("domain_um", "psf_sigma_um", "noise_sigma_V", "time_scale_s", "D_initial_m2_s", "gain_initial"):
        if not np.isfinite(m[key]) or m[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if not 0 < 2*m["electrode_length_um"] < m["domain_um"]:
        raise ValueError("Electrodes must leave a positive gap.")
    if not 0 < m["electrode_width_um"] < m["domain_um"]:
        raise ValueError("Invalid electrode width.")
    if s["truth_grid"] % m["inverse_grid"]:
        raise ValueError("Truth grid must be an integer multiple of inverse grid for the comparison plot.")
    times = np.array(d["nominal_times_s"])
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Nominal times must increase from zero.")
    if d["holdout_nominal_time_s"] not in times or d["holdout_nominal_time_s"] == 0:
        raise ValueError("Choose an existing nonzero time for holdout.")
    if training["steps"] < 1 or d["point_dwell_s"] < 0 or d["min_spacing_um"] <= 0:
        raise ValueError("Invalid training or acquisition settings.")
    count = 4+3*d["points_per_y_band"]+8
    scan_span = (count-1)*d["point_dwell_s"]
    if scan_span >= np.min(np.diff(times)):
        raise ValueError("Successive scans overlap; increase the time separation.")
    if times[-1]+scan_span > m["time_scale_s"]:
        raise ValueError("time_scale_s must cover the full acquisition.")
    if any(not np.isfinite(D) or D <= 0 for D in d["D_prior_m2_s"]):
        raise ValueError("Prior diffusivities must be finite and positive.")


if __name__ == "__main__":
    main()
