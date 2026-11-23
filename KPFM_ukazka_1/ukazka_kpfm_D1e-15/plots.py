"""Static scientific figures, with consistent units and shared colour scales."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 170, "axes.titleweight": "bold"})
COLORS = {"fit": "#ec6a21", "calibration": "#1e78ac", "spatial_validation": "#9255a4"}


def geometry(ax):
    for x in (0, 6):
        ax.add_patch(Rectangle((x, 4), 4, 2, facecolor="#d8dce2", edgecolor="#28384a", zorder=2))
    ax.text(1.7, 4.5, "2 V", ha="center", va="center", zorder=5)
    ax.text(8.3, 4.5, "0 V", ha="center", va="center", zorder=5)
    ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="x [µm]", ylabel="y [µm]")
    ax.set_aspect("equal")


def plan_plot(out, sites, candidates, sensitivity, c):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
    geometry(axes[0]); geometry(axes[1])
    for role, label in (("fit", "Fit: 24 míst"), ("calibration", "Kalibrace: 4 místa"),
                         ("spatial_validation", "Kontrola: 8 míst")):
        points = np.array([[s["x_um"], s["y_um"]] for s in sites if s["role"] == role])
        axes[0].scatter(*points.T, c=COLORS[role], s=30, edgecolor="white", linewidth=.5, zorder=4, label=label)
    axes[0].legend(loc="upper right", fontsize=8)
    for site in sites:
        if site["role"] == "fit":
            axes[0].annotate(site["site_id"][1:], (site["x_um"], site["y_um"]), xytext=(3, 3),
                             textcoords="offset points", fontsize=6, zorder=6)
    sigma = c["model"]["psf_sigma_um"]
    axes[0].add_patch(Circle((8.7, 1.3), 2*sigma, fill=False, color="#28384a"))
    axes[0].text(8.7, .3, "poloměr 2σ", ha="center", fontsize=8)
    axes[0].set_title("1 · Kde měřit")
    sc = axes[1].scatter(*candidates.T, c=sensitivity*1000, cmap="viridis", s=13, zorder=1)
    fig.colorbar(sc, ax=axes[1], shrink=.7, label="RMS ∂V / ∂ln D [mV]")
    axes[1].set_title("Citlivost po konvoluci")
    x = np.linspace(-1.5, 1.5, 400)
    axes[2].plot(x, np.exp(-.5*(x/sigma)**2), color="#1e78ac", lw=2)
    axes[2].axvline(0, color="gray", lw=.7)
    axes[2].axhline(.5, color="gray", lw=.7, ls="--")
    axes[2].set(xlabel="Vzdálenost od středu sondy [µm]", ylabel="Relativní váha PSF", title="Co znamená jeden bod KPFM")
    axes[2].text(.04, .55, f"Gaussovská PSF ve 2D\nσ = {sigma:.2f} µm\nFWHM = {2.355*sigma:.2f} µm\n\nNejmenší odstup = {c['design']['min_spacing_um']:.2f} µm\nŠum = {c['model']['noise_sigma_V']*1000:.0f} mV",
                  transform=axes[2].transAxes, va="top", fontsize=10,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    fig.suptitle(f"KPFM · plán měření · D v rozsahu {min(c['design']['D_prior_m2_s']):.1e}–{max(c['design']['D_prior_m2_s']):.1e} m²/s", fontsize=15)
    fig.savefig(out/"01_plan_mereni.png"); plt.close(fig)


def data_plot(out, surface, fields, rows, times):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    indices = [0, list(times).index(60), -1]
    for col, k in enumerate(indices):
        im = axes[0, col].pcolormesh(surface.faces, surface.faces, fields[k].reshape(surface.n, surface.n),
                                  vmin=0, vmax=2, cmap="turbo", shading="flat", rasterized=True)
        axes[0, col].set_aspect("equal")
        axes[0, col].set_title(f"Latentní potenciál · {times[k]} s")
        axes[0, col].set(xlabel="x [µm]", ylabel="y [µm]")
        points = [r for r in rows if float(r["nominal_time_s"]) == times[k]]
        geometry(axes[1, col])
        sc = axes[1, col].scatter([r["x_um"] for r in points], [r["y_um"] for r in points],
                                  c=[r["voltage_V"] for r in points], vmin=0, vmax=1.5,
                                  cmap="turbo", s=75, edgecolor="black", linewidth=.4, zorder=4)
        axes[1, col].set_title("KPFM: konvoluce + kalibrace + šum")
    fig.colorbar(im, ax=axes[0, :], label="φ [V]", shrink=.8)
    fig.colorbar(sc, ax=axes[1, :], label="Měřený KPFM signál [V]", shrink=.8)
    fig.suptitle("2 · Nezávislá simulace na jemnější mřížce", fontsize=16)
    fig.savefig(out/"02_synteticka_data.png"); plt.close(fig)


def fit_plots(out, basis, network, rows, predictions, history, truth_fields, c, report,
              truth_axis=None, truth_faces=None):
    import torch
    times = np.array(c["design"]["nominal_times_s"])
    with torch.no_grad():
        q = network.clock(torch.as_tensor(times[:, None]/c["model"]["time_scale_s"], dtype=torch.float64)).numpy().ravel()
    reconstructed = basis.fields(q)
    np.savez_compressed(out/"reconstruction.npz", times_s=times, x_um=basis.surface.axis,
                        y_um=basis.surface.axis, faces_um=basis.surface.faces,
                        potential_V=reconstructed.reshape(-1, basis.surface.n, basis.surface.n))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row, time in enumerate((60, 600)):
        k = list(times).index(time)
        fine_n = int(np.sqrt(truth_fields.shape[1]))
        true = truth_fields[k].reshape(fine_n, fine_n)
        pred = reconstructed[k].reshape(basis.surface.n, basis.surface.n)
        from scipy.interpolate import RegularGridInterpolator
        true_coarse = RegularGridInterpolator((truth_axis, truth_axis), true,
                     bounds_error=False, fill_value=None)(basis.surface.xy[:, [1, 0]]).reshape(pred.shape)
        for col, field in enumerate((true, pred, pred-true_coarse)):
            faces = truth_faces if col == 0 else basis.surface.faces
            im = axes[row, col].pcolormesh(faces, faces, field, shading="flat", rasterized=True,
                                       cmap="RdBu_r" if col == 2 else "turbo",
                                       vmin=-.12 if col == 2 else 0, vmax=.12 if col == 2 else 2)
            axes[row, col].set_aspect("equal")
            axes[row, col].set(xlabel="x [µm]", ylabel="y [µm]",
                               title=f"{('Syntetická pravda', 'Rekonstrukce PINN', 'PINN − pravda')[col]} · {time} s")
            fig.colorbar(im, ax=axes[row, col], shrink=.7, label="φ [V]")
    fig.suptitle("3 · Rekonstrukce povrchu z řídkých bodů", fontsize=16)
    fig.savefig(out/"03_rekonstrukce.png"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fit_sites = [r["site_id"] for r in rows if r["role"] == "fit"]
    chosen = list(dict.fromkeys(fit_sites))[::5][:4]
    palette = plt.get_cmap("tab10")
    from pinn import tensors_for_rows
    for j, site in enumerate(chosen):
        mask = np.array([r["site_id"] == site for r in rows])
        part = [r for r, m in zip(rows, mask) if m]
        t = np.array([float(r["t_s"]) for r in part])
        dense = [{**part[0], "t_s": float(v)} for v in np.linspace(0, 602, 150)]
        inputs = tensors_for_rows(basis, dense, c["model"])
        with torch.no_grad():
            v = network.signal(*inputs[:4]).numpy().ravel()
        axes[0, 0].plot([r["t_s"] for r in dense], v, color=palette(j), label=site)
        axes[0, 0].scatter(t, [float(r["voltage_V"]) for r in part], c=[palette(j)], s=20)
    axes[0, 0].axvspan(120, 122, color="gray", alpha=.3, label="Vynechaný čas")
    axes[0, 0].set(xlabel="Skutečný čas [s]", ylabel="KPFM [V]", title="Měřené body a predikce PINN")
    axes[0, 0].legend(fontsize=8)
    train = np.array([bool(int(r["train"])) for r in rows])
    measured = np.array([float(r["voltage_V"]) for r in rows])
    for mask, name, color in ((train, "Trénink", "#1e78ac"), (~train, "Vynechaná data", "#ec6a21")):
        axes[0, 1].scatter(measured[mask], predictions[mask], s=13, alpha=.65, label=name, color=color)
    axes[0, 1].plot([0, 1.5], [0, 1.5], color="black", lw=1)
    axes[0, 1].set(xlabel="Zašuměná data [V]", ylabel="PINN [V]", title="Kontrola mimo trénink")
    axes[0, 1].legend()
    steps = [h["step"] for h in history]
    axes[1, 0].plot(steps, [h["D_m2_s"] for h in history], color="#1e78ac", lw=2, label="Naučené D")
    axes[1, 0].axhline(c["synthetic_only"]["D_true_m2_s"], color="black", ls="--", label="Syntetická pravda")
    axes[1, 0].set(xlabel="Krok Adam", ylabel="D [m²/s]", yscale="log", title="Inverze difuzního koeficientu")
    axes[1, 0].legend()
    axes[1, 1].semilogy(steps, [h["data_loss"] for h in history], label="Data / σ²")
    axes[1, 1].semilogy(steps, [c["training"]["physics_weight"]*h["physics_loss"] for h in history], label="Vážená fyzika")
    axes[1, 1].set(xlabel="Krok Adam", ylabel="Loss", title="Datová a fyzikální část loss")
    axes[1, 1].legend()
    fig.suptitle(f"PINN: D = {report['D_eff_m2_s']:.3e} m²/s · kontrolní RMSE = {report['all_holdout_RMSE_V']*1000:.1f} mV", fontsize=15)
    fig.savefig(out/"04_fit_a_validace.png"); plt.close(fig)
