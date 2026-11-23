"""Plan coordinates without looking at the synthetic truth or noisy data."""
from __future__ import annotations

import csv
import numpy as np


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def choose_sites(basis, model, design):
    # Calibration on both plateaus; PSF mixing is still included there.
    reference = np.array([[1.5, 5.], [2.7, 5.], [7.3, 5.], [8.5, 5.]])
    validation = np.array([[4.6, 4.6], [5.4, 5.4], [3., 3.], [7., 7.],
                           [2., 7.], [8., 3.], [5., 2.], [5., 8.]])
    margin = design.get("candidate_margin_um", .9)
    step = design.get("candidate_step_um", .3)
    axis = np.arange(margin, model["domain_um"]-margin+step/2, step)
    edges = [model["electrode_length_um"], model["domain_um"]-model["electrode_length_um"],
             (model["domain_um"]-model["electrode_width_um"])/2,
             (model["domain_um"]+model["electrode_width_um"])/2]
    extra = [edge+sign*delta for edge in edges for sign in (-1, 1)
             for delta in design.get("extra_edge_distances_um", [])]
    axis = np.unique(np.round(np.r_[axis, extra], 9))
    axis = axis[(axis >= margin) & (axis <= model["domain_um"]-margin)]
    xx, yy = np.meshgrid(axis, axis)
    candidates = np.column_stack((xx.ravel(), yy.ravel()))
    l, r = basis.surface.metal_masks(candidates)
    allowed = ~(l | r)
    allowed &= np.linalg.norm(candidates[:, None]-validation[None], axis=2).min(1) > .38
    candidates = candidates[allowed]
    pool = np.vstack((reference, candidates))
    times = np.array(design["nominal_times_s"])
    times = times[times != design["holdout_nominal_time_s"]]
    steady, modes = basis.observation(pool)
    information, sensitivities = [], []
    for D in design["D_prior_m2_s"]:
        q = D*times/(model["domain_um"]*1e-6)**2
        decay = np.exp(-basis.rates[:, None]*q[None, :])
        signal = steady[:, None]+modes @ decay
        derivative = modes @ (-basis.rates[:, None]*q[None, :]*decay)
        # Unknown log(gain) and offset enter the Fisher matrix as nuisance
        # parameters. Planning gain is a nominal calibration, not truth.
        gain = model["gain_initial"]
        jacobian = np.stack((gain*derivative, gain*signal, np.ones_like(signal)), axis=2)
        jacobian /= model["noise_sigma_V"]
        information.append(np.einsum("nti,ntj->nij", jacobian, jacobian))
        sensitivities.append(np.sqrt(np.mean((gain*derivative)**2, axis=1)))
    information = np.asarray(information)
    total = information[:, :len(reference)].sum(1)+np.eye(3)[None]*1e-6
    selected = []
    blocked = np.linalg.norm(candidates[:, None]-reference[None], axis=2).min(1) < design["min_spacing_um"]
    # Explicit transverse coverage prevents an all-on-axis plan for scalar D.
    band_edges = design.get("transverse_band_edges_um", [0, 3.8, 6.2, 10])
    bands = list(zip(band_edges[:-1], band_edges[1:]))
    for lo, hi in bands:
        for _ in range(design["points_per_y_band"]):
            valid = ~blocked & (candidates[:, 1] >= lo) & (candidates[:, 1] < hi)
            scores = np.linalg.slogdet(total[:, None]+information[:, len(reference):])[1].mean(0)
            scores[~valid] = -np.inf
            k = int(np.argmax(scores))
            if not np.isfinite(scores[k]):
                raise ValueError("Not enough candidates for the spacing and coverage constraints.")
            selected.append(k)
            total += information[:, len(reference)+k]
            blocked |= np.linalg.norm(candidates-candidates[k], axis=1) < design["min_spacing_um"]
    rows = []
    for role, points in (("calibration", reference), ("fit", candidates[selected]),
                         ("spatial_validation", validation)):
        for point in points:
            rows.append({"site_id": f"P{len(rows)+1:02d}", "x_um": float(point[0]),
                         "y_um": float(point[1]), "role": role})
    sensitivity = np.mean(sensitivities, axis=0)[len(reference):]
    return rows, candidates, sensitivity


def make_measurements(surface, sites, model, design, synthetic):
    points = np.array([[row["x_um"], row["y_um"]] for row in sites])
    H = surface.psf(points)
    acquisition = []
    for nominal in design["nominal_times_s"]:
        for i, site in enumerate(sites):
            time = nominal+i*design["point_dwell_s"]
            train = site["role"] != "spatial_validation" and nominal != design["holdout_nominal_time_s"]
            acquisition.append({**site, "nominal_time_s": nominal, "t_s": time,
                                "train": int(train), "time_holdout": int(nominal == design["holdout_nominal_time_s"])})
    times = np.unique([row["t_s"] for row in acquisition]+design["nominal_times_s"])
    fields = surface.simulate(times, synthetic["D_true_m2_s"])
    lookup = {time: i for i, time in enumerate(times)}
    rng = np.random.default_rng(synthetic["noise_seed"])
    measurements, truth = [], []
    for k, row in enumerate(acquisition):
        site = k % len(sites)
        potential = H[site] @ fields[lookup[row["t_s"]]]
        clean = synthetic["gain_true"]*potential+synthetic["offset_true_V"]
        noise = rng.normal(0, model["noise_sigma_V"])
        measurements.append({**row, "voltage_V": clean+noise,
                             "noise_sigma_V": model["noise_sigma_V"]})
        truth.append({"site_id": row["site_id"], "t_s": row["t_s"],
                      "blurred_potential_V": potential, "clean_KPFM_V": clean,
                      "noise_V": noise})
    indices = [lookup[time] for time in design["nominal_times_s"]]
    return measurements, truth, fields[indices]
