"""Time-resolved sensitivity diagnostics; does not change the measurement plan.

Run from F:/PINN:
  .venv/Scripts/python.exe ukazka_kpfm_D1e-15/time_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from physics import Surface, SpatialBasis
from design import read_csv, write_csv

ROOT = Path(__file__).resolve().parent


def sensitivities(basis, points, times, D, gain):
    """Return V, dV/dlnD and dV/dt, using the exact discrete-model derivative."""
    steady, modes = basis.observation(points)
    times = np.asarray(times)
    rate = D/(basis.surface.c["domain_um"]*1e-6)**2
    decay = np.exp(-basis.rates[:, None]*(rate*times[None, :]))
    voltage = gain*(steady[:, None]+modes @ decay)
    dV_dt = gain*modes @ (-basis.rates[:, None]*rate*decay)
    # For constant boundary voltages and a D-independent IC, V depends on D*t.
    # This identity holds through a time-independent spatial PSF as well.
    dV_dlnD = times[None, :]*dV_dt
    return voltage, dV_dlnD, dV_dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=float, help="Illustrative nominal D [m2/s]; not an estimated parameter")
    parser.add_argument("--output", type=Path, default=ROOT/"vystupy")
    args = parser.parse_args()
    c = json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
    args.D = c["diagnostics"]["D_nominal_m2_s"] if args.D is None else args.D
    if not np.isfinite(args.D) or args.D <= 0:
        parser.error("D must be positive and finite")
    model, design = c["model"], c["design"]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    basis = SpatialBasis(Surface(model, model["inverse_grid"]))
    gain = model["gain_initial"]
    # Include time zero explicitly; it has zero D sensitivity despite PSF blur.
    minimum = c["diagnostics"]["curve_min_s"]
    maximum = c["diagnostics"]["curve_max_s"]
    times = np.unique(np.r_[0., np.geomspace(minimum, maximum, 600), design["nominal_times_s"]])
    probes = np.array([[.9, .9], [2., 3.9], [4.025, 5.], [5., 5.], [2., 3.5]])
    labels = ["Levý dolní roh (0,9; 0,9)", "U spodní hrany (2; 3,9)",
              "U čela levé elektrody (4,025; 5)", "Střed mezery (5; 5)", "Dál od hrany (2; 3,5)"]
    voltage, signal_D, signal_t = sensitivities(basis, probes, times, args.D, gain)
    records, peaks = [], []
    for i, label in enumerate(labels):
        k = int(np.argmax(np.abs(signal_D[i])))
        peaks.append({"location": label, "x_um": probes[i, 0], "y_um": probes[i, 1],
                      "peak_time_s": times[k], "peak_abs_dV_dlnD_V": abs(signal_D[i, k]),
                      "peak_inside_600s_protocol": bool(times[k] <= 600),
                      "abs_dV_dlnD_at_600s_V": float(abs(signal_D[i, np.argmin(abs(times-600))]))})
        for j, time in enumerate(times):
            records.append({"location": label, "x_um": probes[i, 0], "y_um": probes[i, 1],
                            "t_s": time, "D_m2_s": args.D, "dV_dlnD_V": signal_D[i, j],
                            "dV_dt_V_per_s": signal_t[i, j]})
    write_csv(out/"time_sensitivity.csv", records)
    # Finite differences independently verify the D/time derivative identity.
    epsilon = 1e-4
    checks = np.array([.1, 1., 30., 120., 600.])
    finite_D = gain*(basis.predict(probes, checks, args.D*np.exp(epsilon))
                     -basis.predict(probes, checks, args.D*np.exp(-epsilon)))/(2*epsilon)
    analytic_D = sensitivities(basis, probes, checks, args.D, gain)[1]
    finite_t = gain*(basis.predict(probes, checks*(1+epsilon), args.D)
                     -basis.predict(probes, checks*(1-epsilon), args.D))/(2*epsilon*checks[None, :])
    analytic_t = sensitivities(basis, probes, checks, args.D, gain)[2]
    np.testing.assert_allclose(analytic_D, finite_D, atol=1e-8, rtol=1e-5)
    np.testing.assert_allclose(analytic_t, finite_t, atol=1e-8, rtol=1e-5)
    assert np.all(signal_D[:, 0] == 0)
    report = {"nominal_D_m2_s": args.D, "planning_gain": gain,
              "initial_dielectric_V": model["initial_dielectric_V"],
              "t0_D_sensitivity_is_zero": True,
              "dV_dlnD_equals_t_times_dV_dt_max_error_V": float(abs(analytic_D-finite_D).max()),
              "location_peaks": peaks,
              "map_is_conditional_on_no_flux_external_boundary": True,
              "diagnostic_only_original_times_not_optimized": True}
    (out/"time_sensitivity_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    from plots import plt, geometry
    from matplotlib.colors import Normalize
    map_times = np.array(c["diagnostics"]["map_times_s"])
    axis = np.unique(np.r_[np.linspace(.5, 9.5, 37), 3.9, 3.975, 4.025, 4.1, 5.9, 5.975, 6.025, 6.1])
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    left, right = basis.surface.metal_masks(points)
    keep = ~(left | right)
    spatial = np.abs(sensitivities(basis, points[keep], map_times, args.D, gain)[1])*1000
    norm = Normalize(0, spatial.max())
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for j, ax in enumerate(axes.ravel()):
        geometry(ax)
        sc = ax.scatter(*points[keep].T, c=spatial[:, j], s=15, cmap="viridis", norm=norm, zorder=1)
        ax.set_title(f"t = {map_times[j]:g} s")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="|∂V / ∂ln D| [mV]", shrink=.8)
    fig.suptitle(f"Citlivost na D v jednotlivých časech · D = {args.D:.1e} m²/s\nStejná 2D PSF, společná barevná škála, vnější okraj bez toku", fontsize=14)
    fig.savefig(out/"05_citlivost_v_case_mapy.png"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for i, label in enumerate(labels):
        line, = axes[0, 0].semilogx(times[1:], np.abs(signal_D[i, 1:])*1000, label=label)
        axes[0, 1].loglog(times[1:], np.maximum(np.abs(signal_t[i, 1:])*1000, 1e-8), color=line.get_color())
    for ax in axes[0]:
        ax.axvline(600, color="gray", ls="--", lw=1)
        ax.set(xlim=(minimum, maximum), xlabel="Čas od zapnutí [s]")
        ax.grid(alpha=.15)
    axes[0, 0].set(ylabel="|∂V / ∂ln D| [mV]", title="Kdy dané místo nejlépe rozlišuje D")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set(ylabel="|∂V / ∂t| [mV/s]", ylim=(1e-7, None), title="Kdy je signál citlivý na chybu času")
    sites = [r for r in read_csv(out/"measurement_plan.csv") if r["role"] == "fit"]
    selected = np.array([[float(r["x_um"]), float(r["y_um"])] for r in sites])
    for D in design["D_prior_m2_s"]:
        derivative = sensitivities(basis, selected, times, D, gain)[1]
        rms = np.sqrt(np.mean(derivative**2, axis=0))*1000
        axes[1, 0].semilogx(times[1:], rms[1:], label=f"D = {D:.0e}")
    for time in design["nominal_times_s"][1:]:
        axes[1, 0].axvline(time, color="gray", lw=.6, alpha=.35)
    axes[1, 0].set(xlabel="Čas [s]", ylabel="RMS ∂V / ∂ln D [mV]", xlim=(minimum, maximum),
                   title="24 míst · závislost na předběžném D")
    axes[1, 0].legend(fontsize=8)
    # Cumulative spatial D information at the ORIGINAL acquisition timestamps;
    # eliminate the shared unknown gain and offset via a Schur complement.
    training_sites = [r for r in read_csv(out/"measurement_plan.csv") if r["role"] != "spatial_validation"]
    total_F = np.zeros((3, 3))
    information = []
    nominal_times = np.array([t for t in design["nominal_times_s"] if t != design["holdout_nominal_time_s"]])
    for nominal in nominal_times:
        for i, row in enumerate(training_sites):
            actual = nominal+i*design["point_dwell_s"]
            signal, dD, _ = sensitivities(basis, [[float(row["x_um"]), float(row["y_um"])]], [actual], args.D, gain)
            jac = np.array([dD[0, 0], signal[0, 0], 1.])/model["noise_sigma_V"]
            total_F += np.outer(jac, jac)
        profiled = total_F[0, 0]-total_F[0, 1:] @ np.linalg.solve(total_F[1:, 1:], total_F[1:, 0])
        information.append(max(profiled, 0))
    axes[1, 1].plot(nominal_times, information, "o-", color="#1e78ac")
    axes[1, 1].set(xlabel="Poslední zahrnutý průchod [s]", ylabel="Profilovaná informace o ln D",
                   title="Původní protokol · přínos dalších průchodů")
    axes[1, 1].text(.03, .95, "Neznámý zisk a offset eliminovány\nČas 120 s zůstává vynechaný\nPředpoklad nezávislého šumu",
                     transform=axes[1, 1].transAxes, va="top", fontsize=9)
    fig.suptitle("Časová citlivost není totéž co rychlost změny signálu", fontsize=15)
    fig.savefig(out/"06_casova_citlivost.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    within = times <= 600
    for i, label in enumerate(labels):
        axes[0].plot(times[within], abs(signal_D[i, within])*1000, label=label)
        axes[1].plot(times[within], (voltage[i, within]-voltage[i, 0])*1000, label=label)
    for ax in axes:
        ax.set(xlabel="Čas od zapnutí [s]", xlim=(0, 600))
        ax.grid(alpha=.15)
    axes[0].set(ylabel="|∂V / ∂ln D| [mV]", title="Citlivost na D v měřeném okně")
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel="V(t) − V(0) [mV]", title="Změna rozmazaného signálu od zapnutí")
    axes[1].axhspan(-model["noise_sigma_V"]*1000, model["noise_sigma_V"]*1000,
                    color="gray", alpha=.2, label="±1σ šumu jednoho odečtu")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"D = {args.D:.0e} m²/s · původní časové okno 0–600 s", fontsize=15)
    fig.savefig(out/"07_detail_0_600s.png"); plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
