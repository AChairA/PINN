"""Inverse 2D KPFM PINN. See README.md for the physical assumptions.

Coordinates: micrometres in files/config, metres in reported D; time: seconds.
Commands: inspect, train, demo, reference. The loader and reference work without PyTorch.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import re
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
EPS0 = 8.8541878128e-12


@dataclass
class Measurement:
    path: Path
    rh: float
    x_um: np.ndarray
    times_s: np.ndarray
    voltage: np.ndarray  # (position, time), including the first numeric row
    time_source: str


def load_measurement(path: Path, config: dict) -> Measurement:
    """Read whitespace/tab exports, decimal commas, optional time headers."""
    path = Path(path)
    match = re.search(r"RH(\d+(?:\.\d+)?)", path.name, re.I)
    if not match:
        raise ValueError(f"{path}: filename must contain RH<number>.")
    lines = [s.strip() for s in path.read_text(encoding="utf-8-sig").splitlines()
             if s.strip() and not s.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{path}: empty file.")
    first = lines[0].replace(",", ".")
    try:
        [float(s) for s in first.split()]
        header = False
    except ValueError:
        header = True
    overrides = config.get("times_s_by_file", {})
    if path.name in overrides:
        times = np.asarray(overrides[path.name], dtype=float)
        source = "config.times_s_by_file"
    elif header:
        pairs = re.findall(r"([-+]?\d+(?:\.\d+)?)\s*(min|s)\b", first, re.I)
        if not pairs:
            raise ValueError(f"{path}: cannot parse time header; set times_s_by_file.")
        times = np.array([float(v) * (60 if u.lower() == "min" else 1)
                          for v, u in pairs])
        source = "header"
    else:
        raise ValueError(f"{path}: missing time header; set explicit times_s_by_file.")
    numeric = "\n".join(lines[int(header):]).replace(",", ".")
    values = np.loadtxt(io.StringIO(numeric), ndmin=2)
    if values.shape[1] != len(times) + 1:
        raise ValueError(f"{path}: {values.shape[1]-1} voltage columns, {len(times)} times.")
    if (len(times) < 2 or not np.isfinite(times).all() or times[0] != 0
            or np.any(np.diff(times) <= 0)):
        raise ValueError(f"{path}: times must start at zero and strictly increase.")
    if len(values) < 3 or not np.isfinite(values).all():
        raise ValueError(f"{path}: need at least 3 finite data rows.")
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError(f"{path}: x must strictly increase.")
    x_um = values[:, 0] + config["scan_x_offset_um"]
    L = config["domain_um"]
    if x_um.min() < 0 or x_um.max() > L:
        raise ValueError(f"{path}: shifted scan is outside [0, {L}] um.")
    return Measurement(path, float(match.group(1)), x_um, times, values[:, 1:], source)


def load_all(data_dir: Path, config: dict) -> list[Measurement]:
    files = list(Path(data_dir).glob("RH*.dat"))
    if not files:
        raise ValueError(f"No RH*.dat files in {data_dir}")
    result = sorted((load_measurement(p, config) for p in files), key=lambda m: m.rh)
    if len({m.rh for m in result}) != len(result):
        raise ValueError("Repeated RH: supply one measurement file per humidity.")
    return result


def validate_config(c: dict):
    for key in ("domain_um", "electrode_length_um", "electrode_width_um",
                "oxide_thickness_nm", "epsilon_r", "D_initial_m2_s"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be positive and finite.")
    L, ell, w = c["domain_um"], c["electrode_length_um"], c["electrode_width_um"]
    if not (2 * ell < L and w < L and 0 <= c["scan_y_um"] <= L):
        raise ValueError("Invalid electrode geometry / scan y.")
    for key in ("left_voltage_V", "right_voltage_V", "backgate_voltage_V",
                "scan_x_offset_um", "scan_y_um"):
        if not np.isfinite(c[key]):
            raise ValueError(f"{key} must be finite.")
    if c["left_voltage_V"] <= c["right_voltage_V"]:
        raise ValueError("This implementation expects left_voltage_V > right_voltage_V.")
    if c["observation"] not in ("affine", "identity"):
        raise ValueError("observation must be affine or identity.")
    if c["initial_condition"] not in ("left_electrode_only", "extrude_first_profile", "electrode_envelope_first_profile"):
        raise ValueError("Unknown initial_condition.")
    if c["initial_condition"] == "electrode_envelope_first_profile":
        if c.get("transverse_decay_um", 0) <= 0:
            raise ValueError("transverse_decay_um must be positive for electrode envelope.")
    if c["outer_boundary"] != "no_flux":
        raise ValueError("Supported outer boundary: no_flux.")
    for key in ("data", "initial", "pde", "electrode", "outer_noflux", "calibration"):
        value = c["loss_weights"].get(key)
        if value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"loss_weights.{key} must be positive and finite.")
    delay = c.get("measurement_time_offset_s", 0.)
    max_delay = c.get("max_measurement_time_offset_s", 0.)
    if not np.isfinite([delay, max_delay]).all() or min(delay, max_delay) < 0 or delay > max_delay:
        raise ValueError("Measurement time offset must lie between zero and its configured maximum.")


def initial_envelope(xy_um, c):
    """Closure for the initial dielectric state; metal values are separate BCs."""
    xy_um = np.asarray(xy_um)
    if c["initial_condition"] == "left_electrode_only":
        return np.zeros(len(xy_um))
    if c["initial_condition"] == "electrode_envelope_first_profile":
        distance = np.maximum(np.abs(xy_um[:, 1]-c["domain_um"]/2)
                              - c["electrode_width_um"]/2, 0)
        return np.exp(-.5*(distance/c["transverse_decay_um"])**2)
    return np.ones(len(xy_um))


class Geometry:
    """Square minus the two finite rectangular metal electrodes."""
    def __init__(self, c):
        self.L = c["domain_um"]
        self.ell = c["electrode_length_um"] / self.L
        self.lo = (1 - c["electrode_width_um"] / self.L) / 2
        self.hi = 1 - self.lo

    def metal_masks(self, xy):
        band = (xy[:, 1] >= self.lo) & (xy[:, 1] <= self.hi)
        return band & (xy[:, 0] <= self.ell), band & (xy[:, 0] >= 1 - self.ell)

    def dielectric(self, xy):
        a, b = self.metal_masks(xy)
        return ~(a | b)

    def interior(self, rng, n, time_zero=False, gap_fraction=0.):
        collected = []
        count = 0
        while count < n:
            p = rng.random((max(n, 32), 3))
            p = p[self.dielectric(p[:, :2])]
            collected.append(p)
            count += len(p)
        p = np.concatenate(collected)[:n]
        # Resolve the small gap where the measured transient carries most of
        # the information about D; keep uniform coverage elsewhere as well.
        ngap = int(n * gap_fraction)
        if ngap:
            p[:ngap, 0] = self.ell + rng.random(ngap) * (1 - 2 * self.ell)
            p[:ngap, 1] = self.lo + rng.random(ngap) * (self.hi - self.lo)
        # Initial transients can be much shorter than the recording duration.
        # Keep half the times uniform and cluster half near t=0.
        p[:n//2, 2] **= 3
        if time_zero:
            p[:, 2] = 0
        return p

    def electrodes(self, rng, n):
        """Sample all three exposed sides, weighted by their lengths."""
        result = []
        for right in (False, True):
            p = rng.random((n, 3))
            side = rng.random(n) * (2 * self.ell + self.hi - self.lo)
            bottom = side < self.ell
            top = (side >= self.ell) & (side < 2 * self.ell)
            tip = ~(bottom | top)
            p[:, 0] *= self.ell
            p[bottom, 1] = self.lo
            p[top, 1] = self.hi
            p[tip, 0] = self.ell
            p[tip, 1] = self.lo + rng.random(tip.sum()) * (self.hi - self.lo)
            if right:
                p[:, 0] = 1 - p[:, 0]
            result.append(p)
        return result

    def outer(self, rng, n):
        points, normals = [], []
        for dim in (0, 1):
            for value in (0., 1.):
                p = rng.random((n, 3))
                p[:, dim] = value
                p = p[self.dielectric(p[:, :2])]
                normal = np.zeros((len(p), 2))
                normal[:, dim] = 2 * value - 1
                points.append(p)
                normals.append(normal)
        return np.concatenate(points), np.concatenate(normals)


def summarize(m, config):
    g = Geometry(config)
    xy = np.column_stack((m.x_um / g.L, np.full(len(m.x_um), config["scan_y_um"] / g.L)))
    left, right = g.metal_masks(xy)
    return {
        "file": m.path.name, "RH_percent": m.rh, "n_positions": len(m.x_um),
        "x_absolute_um": [float(m.x_um.min()), float(m.x_um.max())],
        "times_s": m.times_s.tolist(), "time_source": m.time_source,
        "voltage_range_V": [float(m.voltage.min()), float(m.voltage.max())],
        "left_plateau_V": np.median(m.voltage[left], axis=0).tolist() if left.any() else None,
        "right_plateau_V": np.median(m.voltage[right], axis=0).tolist() if right.any() else None,
    }


def make_model(c, t_max, width, depth, device):
    import torch
    from torch import nn

    class PINN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            # Global coordinates plus local gap/early-time coordinates reduce
            # the spectral bias of tanh for a narrow gap and fast transients.
            nin = 6
            for _ in range(depth):
                layer = nn.Linear(nin, width)
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                layers.extend((layer, nn.Tanh()))
                nin = width
            layers.append(nn.Linear(nin, 1))
            self.net = nn.Sequential(*layers)
            # Dimensionless alpha = D * T / L^2; positivity via exp.
            alpha = c["D_initial_m2_s"] * t_max / (c["domain_um"] * 1e-6)**2
            self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))
            self.log_gain = nn.Parameter(torch.tensor(0.), requires_grad=c["observation"] == "affine")
            self.offset = nn.Parameter(torch.tensor(0.), requires_grad=c["observation"] == "affine")

        @property
        def alpha(self):
            return self.log_alpha.exp()

        @property
        def gain(self):
            return self.log_gain.exp()

        def forward(self, p):
            local_xy = (p[:, :2] - .5) * 10
            early_time = 2 * torch.log1p(100*p[:, 2:3]) / math.log(101) - 1
            return self.net(torch.cat((2*p-1, local_xy, early_time), dim=1))

        def volts(self, u):
            return c["right_voltage_V"] + (c["left_voltage_V"] - c["right_voltage_V"]) * u

        def observed(self, u):
            return self.gain * self.volts(u) + self.offset

    return PINN().to(device)


def derivatives(model, p):
    import torch
    u = model(p)
    grad = torch.autograd.grad(u.sum(), p, create_graph=True)[0]
    uxx = torch.autograd.grad(grad[:, 0].sum(), p, create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(grad[:, 1].sum(), p, create_graph=True)[0][:, 1:2]
    return grad[:, 2:3] - model.alpha * (uxx + uyy)


def train_one(m, c, args, output):
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested, but unavailable.")
    rng = np.random.default_rng(args.seed)
    g = Geometry(c)
    scale = c["left_voltage_V"] - c["right_voltage_V"]
    reference = None
    if args.reference_d:
        from kpfm_reference import Reference
        print(f"RH{m.rh:g}: fitting independent numerical reference ...", flush=True)
        reference = Reference(m, c, args.reference_grid).fit(args.holdout_time_index)
        c = dict(c)
        c["D_initial_m2_s"] = reference["D_eff_m2_s"]
        c["measurement_time_offset_s"] = reference["measurement_time_offset_s"]
        print(f"Reference D={reference['D_eff_m2_s']:.5g} m2/s, "
              f"RMSE={reference['fit_RMSE_V']:.4g} V", flush=True)
    delay = c.get("measurement_time_offset_s", 0.)
    T = float(m.times_s[-1] + delay)
    model = make_model(c, T, args.width, args.depth, args.device)

    def tensor(a, grad=False):
        return torch.tensor(a, dtype=torch.float32, device=args.device, requires_grad=grad)

    # Order: position first, time second, exactly as voltage.ravel().
    x, tt = np.meshgrid(m.x_um / g.L, (m.times_s + delay) / T, indexing="ij")
    p_all = np.column_stack((x.ravel(), np.full(x.size, c["scan_y_um"] / g.L), tt.ravel()))
    v_all = m.voltage.reshape(-1, 1)
    ml, mr = g.metal_masks(p_all[:, :2])
    dielectric = ~(ml | mr)
    # Hold out entire nonzero time profiles when requested, not random neighbours.
    hold_times = set(args.holdout_time_index)
    if any(i <= 0 or i >= len(m.times_s) for i in hold_times):
        raise ValueError("Holdout indices must refer to nonzero time columns.")
    held = np.tile(np.isin(np.arange(len(m.times_s)), list(hold_times)), len(m.x_um))
    fit = dielectric & ~held
    if not fit.any():
        raise ValueError("No training data in dielectric; check scan coordinates.")
    cal = (ml | mr) & ~held
    if c["observation"] == "affine" and (not (ml & ~held).any() or not (mr & ~held).any()):
        raise ValueError("Affine calibration needs measured points on both electrodes.")
    known = np.where(ml, c["left_voltage_V"], c["right_voltage_V"])
    if c["observation"] == "affine":
        vleft = float(np.median(v_all[ml & ~held]))
        vright = float(np.median(v_all[mr & ~held]))
        gain = (vleft - vright) / scale
        if gain <= 0:
            raise ValueError("Left/right plateaus incompatible with positive affine gain.")
        with torch.no_grad():
            model.log_gain.fill_(math.log(gain))
            model.offset.fill_(vright - gain * c["right_voltage_V"])
    if reference is not None:
        with torch.no_grad():
            model.log_gain.fill_(math.log(reference["gain"]))
            model.offset.fill_(reference["offset_V"])
    p_fit, v_fit = tensor(p_all[fit]), tensor(v_all[fit])
    v_cal, known_cal = tensor(v_all[cal]), tensor(known[cal, None])

    def batch():
        interior = tensor(g.interior(rng, args.collocation, gap_fraction=args.gap_fraction), True)
        initial_np = g.interior(rng, args.initial_points, time_zero=True)
        # With left_electrode_only the envelope is zero: dielectric starts at
        # backgate potential independently of the first measured profile.
        target = np.interp(initial_np[:, 0] * g.L, m.x_um, m.voltage[:, 0])[:, None]
        envelope = initial_envelope(initial_np[:, :2]*g.L, c)[:, None]
        initial = tensor(initial_np)
        left, right = (tensor(p) for p in g.electrodes(rng, args.boundary_points))
        outer, normals = g.outer(rng, args.boundary_points)
        return interior, initial, tensor(target), left, right, tensor(outer, True), tensor(normals), tensor(envelope)

    def losses(b, physics_weight=1.):
        interior, initial, target, left, right, outer, normals, envelope = b
        data = ((model.observed(model(p_fit)) - v_fit) / scale).square().mean()
        reference = model.gain * c["backgate_voltage_V"] + model.offset
        initial_target = reference + envelope * (target - reference)
        ic = ((model.observed(model(initial)) - initial_target) / scale).square().mean()
        pde = derivatives(model, interior).square().mean()
        electrode = (model(left) - 1).square().mean() + model(right).square().mean()
        grad = torch.autograd.grad(model(outer).sum(), outer, create_graph=True)[0][:, :2]
        noflux = (grad * normals).sum(dim=1).square().mean()
        calibration = (((model.gain * known_cal + model.offset - v_cal) / scale).square().mean()
                       if len(v_cal) else torch.zeros((), device=args.device))
        parts = dict(data=data, initial=ic, pde=pde, electrode=electrode,
                     outer_noflux=noflux, calibration=calibration)
        total = sum(c["loss_weights"][key] * value * (physics_weight if key == "pde" else 1.)
                    for key, value in parts.items())
        return total, parts

    # D gets a separate learning rate; physical/NN parameters update together.
    other = [p for name, p in model.named_parameters() if name != "log_alpha" and p.requires_grad]
    opt = torch.optim.Adam([{"params": other, "lr": args.lr},
                            {"params": [model.log_alpha], "lr": args.parameter_lr}])
    output.mkdir(parents=True, exist_ok=True)
    history = []
    start = time.perf_counter()
    warmup = min(args.parameter_warmup, args.steps // 2)
    for step in range(1, args.steps + 1):
        # Fit the field with a fixed D before differentiating the objective
        # with respect to D. Otherwise initial NN curvature can drive D to zero.
        model.log_alpha.requires_grad_(reference is None and step > warmup)
        opt.zero_grad(set_to_none=True)
        b = batch()
        # First learn the observed transient. Gradually introduce the PDE,
        # rather than rewarding a time-independent field at initialization.
        physics_weight = min(1., max(0., (step - warmup/2) / max(1., warmup/2)))
        total, parts = losses(b, physics_weight)
        if not torch.isfinite(total):
            raise RuntimeError(f"Nonfinite loss at step {step}; reduce learning rate.")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {"step": step, "total": total.item(),
                   **{k: v.item() for k, v in parts.items()},
                   "D_m2_s": model.alpha.item() * (g.L * 1e-6)**2 / T,
                   "gain": model.gain.item(), "offset_V": model.offset.item()}
            history.append(row)
            print(f"RH{m.rh:g} seed={args.seed} step={step} loss={row['total']:.4g} "
                  f"D={row['D_m2_s']:.4g} m2/s", flush=True)
    if args.lbfgs_steps:
        fixed = batch()  # fixed collocation set for deterministic line search
        opt2 = torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad],
                                max_iter=args.lbfgs_steps, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad(set_to_none=True)
            # Input gradients are not optimizer parameters; clear between closures.
            for p in fixed:
                if p.requires_grad:
                    p.grad = None
            total, _ = losses(fixed)
            if not torch.isfinite(total):
                raise RuntimeError("Nonfinite LBFGS objective.")
            total.backward()
            return total

        opt2.step(closure)
        total, parts = losses(fixed)
        history.append({"step": args.steps + args.lbfgs_steps, "total": total.item(),
                        **{k: v.item() for k, v in parts.items()},
                        "D_m2_s": model.alpha.item() * (g.L * 1e-6)**2 / T,
                        "gain": model.gain.item(), "offset_V": model.offset.item()})
    check = diagnostics(model, g, tensor, c, args)
    with torch.no_grad():
        u = model(tensor(p_all)).cpu().numpy().ravel()
        u[ml], u[mr] = 1, 0  # metal interiors are outside the PDE domain
        phi = c["right_voltage_V"] + scale * u
        pred = model.gain.item() * phi + model.offset.item()
    err = pred - v_all.ravel()
    capacitance = EPS0 * c["epsilon_r"] / (c["oxide_thickness_nm"] * 1e-9)
    # Report only the effective capacitor charge on dielectric, not electrode charge.
    sigma = capacitance * (phi - c["backgate_voltage_V"])
    sigma[~dielectric] = np.nan
    sigma0 = sigma.reshape(m.voltage.shape)[:, :1]
    delta = (sigma.reshape(m.voltage.shape) - sigma0).ravel()
    records = np.column_stack((p_all[:, 0] * g.L, p_all[:, 1] * g.L, p_all[:, 2] * T-delay, p_all[:, 2] * T,
                               v_all.ravel(), pred, phi, sigma, delta, held.astype(int), dielectric.astype(int)))
    np.savetxt(output / "profiles.csv", records, delimiter=",", comments="",
               header="x_um,y_um,t_s,t_sim_s,V_measured,V_KPFM_fit,V_model,sigma_eff_C_m2,delta_sigma_eff_C_m2,heldout,is_dielectric")
    D = model.alpha.item() * (g.L * 1e-6)**2 / T
    warnings = []
    if reference and reference["D_near_search_bound"]:
        warnings.append("numerical_reference_D_at_search_bound: report a bound, not a resolved diffusion coefficient")
    if reference and reference.get("time_offset_at_upper_bound"):
        warnings.append("measurement_time_offset_at_upper_bound: inspect first-profile residual and missing instantaneous electrostatic field")
    if D / c["D_initial_m2_s"] < .01:
        warnings.append("D_collapsed_below_1_percent_of_initial_guess: inspect weak temporal sensitivity, model mismatch and optimization")
    if D / c["D_initial_m2_s"] > 100:
        warnings.append("D_exceeds_100_times_initial_guess: test parameter stability and time resolution")
    # These flags diagnose extreme motion of the inverse parameter; they are
    # not bounds on D or an identifiability test.
    summary = {"RH_percent": m.rh, "seed": args.seed, "D_eff_m2_s": D,
               "D_eff_um2_s": D * 1e12, "gain": model.gain.item(),
               "offset_V": model.offset.item(), "capacitance_F_m2": capacitance,
               "measurement_time_offset_s": delay,
               "oxide_thickness_nm_assumed": c["oxide_thickness_nm"],
               "training_RMSE_V": float(np.sqrt(np.mean(err[fit]**2))),
               "calibration_RMSE_V": float(np.sqrt(np.mean(err[cal]**2))) if cal.any() else None,
               "heldout_RMSE_V": float(np.sqrt(np.mean(err[held & dielectric]**2))) if held.any() else None,
               "diagnostics": check, "elapsed_s": time.perf_counter() - start,
               "parameter_warmup_steps": warmup,
               "status": "conditional_model_fit_not_identifiability_certificate",
               "warnings": warnings,
               "D_estimation_method": "independent_numerical_inverse_fixed_in_PINN" if reference else "joint_PINN_inverse",
               "numerical_reference": reference,
               "assumptions": ["local_capacitance", "constant_isotropic_D_per_RH",
                               c["initial_condition"], "no_flux_outer_square",
                               "constant_affine_KPFM_calibration_per_RH"],
               "source": summarize(m, c)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "loss.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    resolved_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (output / "run_config.json").write_text(json.dumps({"physics": c, "training": resolved_args,
                    "versions": {"torch": torch.__version__, "numpy": np.__version__}}, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "physics_config": c,
                "t_max_s": T, "width": args.width, "depth": args.depth}, output / "model.pt")
    save_plots_and_fields(model, m, g, c, tensor, output, history, phi, pred, capacitance)
    return summary


def diagnostics(model, g, tensor, c, args):
    import torch
    # Independent dense grid; not a reused training collocation batch.
    s = np.linspace(.005, .995, 29)
    x, y, t = np.meshgrid(s, s, np.linspace(0, 1, 7), indexing="ij")
    p = np.column_stack((x.ravel(), y.ravel(), t.ravel()))
    p = p[g.dielectric(p[:, :2])]
    residual = []
    for part in np.array_split(p, max(1, math.ceil(len(p) / 512))):
        r = derivatives(model, tensor(part, True)).detach().cpu().numpy()
        residual.extend(r.ravel())
    rng = np.random.default_rng(args.seed + 100_003)
    left, right = g.electrodes(rng, 1024)
    outer, normal = g.outer(rng, 512)
    with torch.no_grad():
        bc = torch.cat((model(tensor(left)) - 1, model(tensor(right)))).cpu().numpy()
    p = tensor(outer, True)
    grad = torch.autograd.grad(model(p).sum(), p)[0][:, :2]
    flux = (grad * tensor(normal)).sum(1).detach().cpu().numpy()
    scale = c["left_voltage_V"] - c["right_voltage_V"]
    return {"pde_dimensionless_RMS": float(np.sqrt(np.mean(np.square(residual)))),
            "pde_dimensionless_max_abs": float(np.max(np.abs(residual))),
            "electrode_RMSE_V": float(scale * np.sqrt(np.mean(bc**2))),
            "outer_normal_derivative_dimensionless_RMS": float(np.sqrt(np.mean(flux**2))),
            "pde_check_points": len(residual)}


def save_plots_and_fields(model, m, g, c, tensor, output, history, phi, pred, capacitance):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, t in enumerate(m.times_s):
        line, = axes[0].plot(m.x_um, pred.reshape(m.voltage.shape)[:, j], label=f"{t:g} s")
        axes[0].plot(m.x_um[::8], m.voltage[::8, j], ".", color=line.get_color(), ms=3)
    axes[0].set(xlabel="x [um]", ylabel="KPFM [V]", title=f"RH {m.rh:g}%: lines PINN, dots data")
    axes[0].legend(fontsize=8)
    for key in ("data", "initial", "pde", "electrode", "outer_noflux", "calibration"):
        axes[1].semilogy([r["step"] for r in history], [max(r[key], 1e-15) for r in history], label=key)
    axes[1].set(xlabel="Optimizer step", ylabel="Unweighted loss")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "fit_and_loss.png", dpi=180)
    plt.close(fig)
    s = np.linspace(0, 1, 101)
    xx, yy = np.meshgrid(s, s, indexing="xy")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    ml, mr = g.metal_masks(xy)
    fields = []
    for t in m.times_s:
        delay = c.get("measurement_time_offset_s", 0.)
        p = np.column_stack((xy, np.full(len(xy), (t+delay) / (m.times_s[-1]+delay))))
        with torch.no_grad():
            u = model(tensor(p)).cpu().numpy().ravel()
        u[ml], u[mr] = 1, 0
        field = (c["right_voltage_V"] + (c["left_voltage_V"] - c["right_voltage_V"]) * u).reshape(xx.shape)
        fields.append(field)
    fields = np.array(fields)
    sigma = capacitance * (fields - c["backgate_voltage_V"])
    sigma[:, (ml | mr).reshape(xx.shape)] = np.nan
    np.savez_compressed(output / "fields.npz", x_um=xx*g.L, y_um=yy*g.L, times_s=m.times_s,
                        simulation_times_s=m.times_s+c.get("measurement_time_offset_s", 0.),
                        phi_V=fields, sigma_eff_C_m2=sigma,
                        dielectric_mask=g.dielectric(xy).reshape(xx.shape))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, j in zip(axes, [0, -1]):
        im = ax.imshow(fields[j], origin="lower", extent=(0, g.L, 0, g.L),
                       vmin=c["right_voltage_V"], vmax=c["left_voltage_V"], cmap="viridis")
        ax.axhline(c["scan_y_um"], color="white", lw=.6, ls="--")
        ax.set(xlabel="x [um]", ylabel="y [um]", title=f"Conditional 2D reconstruction, t={m.times_s[j]:g} s")
        fig.colorbar(im, ax=ax, label="Model potential [V]")
    fig.tight_layout()
    fig.savefig(output / "potential_maps.png", dpi=180)
    plt.close(fig)


def make_demo(config):
    """Independent finite-difference solution on the 2D electrode geometry.

    The steady field is obtained from a sparse finite-difference Laplace solve;
    the demo exercises the pipeline but does not certify inverse D recovery.
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve, expm_multiply
    c = dict(config)
    c["initial_condition"] = "extrude_first_profile"
    c["measurement_time_offset_s"] = 0.
    c["max_measurement_time_offset_s"] = 0.
    c["D_initial_m2_s"] = 1e-14
    c["scan_y_um"] = 5.
    g = Geometry(c)
    n = 51
    s = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(s, s, indexing="xy")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    ml, mr = g.metal_masks(xy)
    metal = ml | mr
    ids = np.full(n*n, -1, dtype=int)
    ids[~metal] = np.arange((~metal).sum())
    A = lil_matrix(((~metal).sum(), (~metal).sum()))
    b = np.zeros((~metal).sum())
    h = 1/(n-1)
    # Reflecting ghost cells impose Neumann conditions on the outer square.
    for k in np.flatnonzero(~metal):
        j, i = divmod(k, n)
        neighbours = [(j, i-1 if i else 1), (j, i+1 if i<n-1 else n-2),
                      (j-1 if j else 1, i), (j+1 if j<n-1 else n-2, i)]
        row = ids[k]
        A[row, row] = -4/h**2
        for jj, ii in neighbours:
            q = jj*n+ii
            if metal[q]:
                b[row] += float(ml[q])/h**2
            else:
                A[row, ids[q]] += 1/h**2
    A = A.tocsr()
    steady = spsolve(A, -b)
    coord = xy[~metal, 0]
    gap_coordinate = (coord - g.ell) / (1 - 2*g.ell)
    initial = np.clip(1-gap_coordinate, 0, 1)
    # A linear gap profile is almost stationary and carries little information
    # about D. Add a resolvable transient, zero at both metal tips.
    in_gap = (gap_coordinate > 0) & (gap_coordinate < 1)
    initial[in_gap] += .3*np.sin(np.pi*gap_coordinate[in_gap])
    D = 2e-14
    times = np.array([0, 30, 60, 150, 300, 450, 600.])
    voltages = []
    gain, offset = .72, .015
    center = n//2
    xs = np.linspace(2, 8, 129)
    for t in times:
        state = steady + expm_multiply(A * (D*t/(g.L*1e-6)**2), initial-steady)
        full = np.zeros(n*n)
        full[ml] = 1
        full[~metal] = state
        line = full.reshape(n, n)[center]
        voltages.append(gain*(c["right_voltage_V"] +
                        (c["left_voltage_V"]-c["right_voltage_V"])*np.interp(xs/g.L, s, line))+offset)
    return Measurement(Path("synthetic_RH50.dat"), 50, xs, times,
                       np.array(voltages).T, "synthetic_finite_difference"), c, D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "train", "demo", "reference"])
    parser.add_argument("--config", type=Path, default=ROOT / "pinn_config.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "pinn")
    parser.add_argument("--rh", nargs="+", type=float, help="Select humidities; default all.")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--collocation", type=int, default=512)
    parser.add_argument("--boundary-points", type=int, default=96, help="Points per electrode / outer side.")
    parser.add_argument("--initial-points", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--parameter-lr", type=float, default=2e-3)
    parser.add_argument("--parameter-warmup", type=int, default=2000,
                        help="Keep D fixed initially; capped at half the Adam steps.")
    parser.add_argument("--gap-fraction", type=float, default=.5,
                        help="Fraction of PDE points focused in the 2x2 um gap.")
    parser.add_argument("--reference-d", action="store_true",
                        help="Estimate D with an independent numerical inverse, then hold it fixed in PINN.")
    parser.add_argument("--reference-grid", type=int, default=31)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", nargs="+", type=int, help="Independent fits for parameter stability.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--holdout-time-index", nargs="*", type=int, default=[])
    args = parser.parse_args()
    try:
        c = json.loads(args.config.read_text(encoding="utf-8"))
        validate_config(c)
        for key in ("steps", "width", "depth", "collocation", "boundary_points",
                    "initial_points", "threads", "log_every"):
            if getattr(args, key) <= 0:
                raise ValueError(f"--{key.replace('_', '-')} must be positive.")
        if args.lbfgs_steps < 0 or args.lr <= 0 or args.parameter_lr <= 0:
            raise ValueError("Optimizer rates must be positive; LBFGS steps nonnegative.")
        if args.parameter_warmup < 0 or not 0 <= args.gap_fraction < 1:
            raise ValueError("Warmup must be nonnegative; gap fraction must be in [0, 1).")
        if args.command == "demo":
            m, c, truth = make_demo(c)
            measurements = [m]
        else:
            measurements = load_all(args.data_dir, c)
            truth = None
        if args.rh:
            selected = set(args.rh)
            missing = selected - {m.rh for m in measurements}
            if missing:
                raise ValueError(f"Missing RH values: {sorted(missing)}")
            measurements = [m for m in measurements if m.rh in selected]
        if args.command == "inspect":
            report = [summarize(m, c) for m in measurements]
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "data_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            for r in report:
                print(f"{r['file']}: {r['n_positions']} x {len(r['times_s'])}, "
                      f"x={r['x_absolute_um']} um, t={r['times_s']} s ({r['time_source']})")
            return
        if args.command == "reference":
            from kpfm_reference import Reference
            args.output.mkdir(parents=True, exist_ok=True)
            rows = []
            for m in measurements:
                result = Reference(m, c, args.reference_grid).fit(args.holdout_time_index)
                rows.append({"RH_percent": m.rh, **result})
                print(f"RH{m.rh:g}: D={result['D_eff_m2_s']:.6g} m2/s, "
                      f"RMSE={result['fit_RMSE_V']:.5g} V, bound={result['D_near_search_bound']}", flush=True)
            (args.output / "reference_results.json").write_text(json.dumps(
                {"physics": c, "holdout_time_index": args.holdout_time_index, "results": rows}, indent=2), encoding="utf-8")
            with (args.output / "reference_results.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            return
        rows = []
        for m in measurements:
            for seed in args.seeds or [args.seed]:
                args.seed = seed
                folder = args.output / f"RH{m.rh:g}" / f"seed{seed}"
                row = train_one(m, c, args, folder)
                if truth is not None:
                    row["synthetic_D_true_m2_s"] = truth
                    row["synthetic_D_relative_error"] = abs(row["D_eff_m2_s"] / truth - 1)
                    if row["synthetic_D_relative_error"] > .2:
                        row["warnings"].append("synthetic_D_recovery_failed_20_percent_tolerance")
                    (folder / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                rows.append(row)
        (args.output / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved {len(rows)} fit(s) in {args.output}")
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
