"""Create 2D maps and time profiles for every RH from saved numerical fits.

Usage: python plot_kpfm.py
The saved physics configuration, fitted parameters and grid are reused exactly.
No neural network is trained and no measured 2D data are inferred from images.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from kpfm_pinn import ROOT, EPS0, load_all, validate_config
from kpfm_reference import Reference


def time_label(t):
    return f"{t/60:g} min" if t >= 60 and t % 30 == 0 else f"{t:g} s"


def parameter_note(result):
    if result["D_near_search_bound"]:
        return "D nerozlišené (optimum na mezi hledání)"
    return f"D_eff = {result['D_eff_m2_s']:.3g} m²/s"


def decorate_map(ax, c, mask_metal=False):
    L = c["domain_um"]
    ell = c["electrode_length_um"]
    width = c["electrode_width_um"]
    for x0 in (0, L-ell):
        ax.add_patch(Rectangle((x0, (L-width)/2), ell, width,
                               facecolor="white" if mask_metal else "none",
                               edgecolor="black", linewidth=.8))
    ax.set(xlim=(0, L), ylim=(0, L), xlabel="x [µm]", ylabel="y [µm]")
    ax.set_aspect("equal")


def map_image(ax, field, case, limits, cmap):
    c, m = case["config"], case["measurement"]
    L = c["domain_um"]
    spacing = L/(field.shape[0]-1)
    masked = np.isnan(field).any()
    if masked:
        # Fill hidden metal nodes for interpolation only, then cover the exact
        # rectangles. NaN interpolation would expand the blank metal region
        # by a whole grid cell beyond the electrode outline.
        from scipy.ndimage import distance_transform_edt
        nearest = distance_transform_edt(np.isnan(field), return_distances=False, return_indices=True)
        field = field[tuple(nearest)]
    # Bilinear interpolation is display only; NPZ retains the actual grid.
    im = ax.imshow(field, origin="lower", extent=(-spacing/2, L+spacing/2, -spacing/2, L+spacing/2),
                   interpolation="bilinear", cmap=cmap, vmin=limits[0], vmax=limits[1])
    decorate_map(ax, c, mask_metal=masked)
    ax.plot([m.x_um[0], m.x_um[-1]], [c["scan_y_um"]]*2,
            color="white", linestyle="--", linewidth=.9)
    return im


def save_maps(case, key, limits, label, title, filename, cmap="viridis"):
    fields = case["fields"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.7), layout="constrained")
    for ax, t, field in zip(axes.ravel(), fields["times_s"], fields[key]):
        im = map_image(ax, field, case, limits, cmap)
        delay = case["result"].get("measurement_time_offset_s", 0.)
        ax.set_title(f"Měření: {time_label(t)}\nOd zapnutí: {t+delay:.2f} s", fontsize=10)
    last = axes.ravel()[-1]
    last.axis("off")
    short_parameter = parameter_note(case["result"]).replace(" (optimum", "\n(optimum")
    initial_note = ("Počátek: pouze levá elektroda" if case["config"]["initial_condition"] == "left_electrode_only"
                    else "Počáteční řez rozšířen ve směru y")
    note = (f"RH {case['measurement'].rh:g} %\n{short_parameter}\n\n"
            "Numerický transportní model\nPodmíněná 2D rekonstrukce\n"
            f"Mřížka {case['result']['grid']} × {case['result']['grid']}\n\n"
            "Černě: obrys elektrod\nBíle: měřený řez y = 5 µm\n"
            f"{initial_note}\nProdleva měření: {delay:.2f} s\n"
            "Společná barevná stupnice všech RH")
    last.text(.02, .95, note, transform=last.transAxes, va="top", fontsize=9.3, linespacing=1.4)
    fig.colorbar(im, ax=axes.ravel()[:-1].tolist(), shrink=.86, label=label, pad=.02)
    fig.suptitle(f"RH {case['measurement'].rh:g} % — {title}", fontsize=16)
    fig.savefig(case["folder"] / filename, dpi=190)
    plt.close(fig)


def save_cuts(case, limits):
    m, c, fields = case["measurement"], case["config"], case["fields"]
    pred = case["prediction"]
    colors = plt.get_cmap("viridis")(np.linspace(.08, .93, len(m.times_s)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    ix = np.argmin(abs(fields["x_um"][0] - c["domain_um"]/2))
    for j, (t, color) in enumerate(zip(m.times_s, colors)):
        label = time_label(t)
        axes[0].plot(m.x_um, pred[:, j], color=color, lw=1.8, label=label)
        axes[0].plot(m.x_um, m.voltage[:, j], color=color, lw=.7, alpha=.35)
        axes[0].plot(m.x_um[::6], m.voltage[::6, j], ".", color=color, ms=3)
        axes[1].plot(fields["y_um"][:, ix], fields["V_KPFM"][j, :, ix],
                     color=color, lw=1.8)
        axes[2].plot(m.x_um, m.voltage[:, j]-pred[:, j], color=color, lw=1)
    axes[0].set(title=f"Podélný řez: y = {c['scan_y_um']:g} µm",
                xlabel="x [µm]", ylabel="KPFM [V]", ylim=limits)
    axes[0].legend(title="Čas", fontsize=8, title_fontsize=9, loc="upper right")
    axes[1].set(title=f"Příčný řez modelu: x = {fields['x_um'][0, ix]:g} µm",
                xlabel="y [µm]", ylabel="Modelový KPFM signál [V]", ylim=limits)
    axes[1].axvline(c["scan_y_um"], color=".6", lw=.8, ls="--")
    axes[2].set(title="Odchylka podélného řezu", xlabel="x [µm]",
                ylabel="Měření − model [V]")
    axes[2].axhline(0, color=".5", lw=.8)
    for ax in axes:
        ax.grid(alpha=.18)
    for ax in (axes[0], axes[2]):
        ax.axvspan(m.x_um[0], c["electrode_length_um"], color=".5", alpha=.08)
        ax.axvspan(c["domain_um"]-c["electrode_length_um"], m.x_um[-1], color=".5", alpha=.08)
    delay = case["result"].get("measurement_time_offset_s", 0.)
    fig.suptitle(f"RH {m.rh:g} % — {parameter_note(case['result'])} · prodleva {delay:.2f} s", fontsize=14)
    fig.legend(handles=[Line2D([], [], color=".3", marker=".", ls="", label="Měřené body"),
                        Line2D([], [], color=".3", lw=1.8, label="Numerický model")],
               loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(case["folder"] / "rezy.png", dpi=190)
    plt.close(fig)


def save_overviews(cases, output, limits):
    columns = 4
    rows = math.ceil((len(cases)+1)/columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, rows*3.9), squeeze=False, layout="constrained")
    used = []
    for ax, case in zip(axes.ravel(), cases):
        im = map_image(ax, case["fields"]["V_KPFM"][-1], case, limits, "viridis")
        m = case["measurement"]
        ax.set_title(f"RH {m.rh:g} % · t = {time_label(m.times_s[-1])}")
        used.append(ax)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    axes.ravel()[len(cases)].text(.02, .95,
        "Modelový KPFM signál\n\nPoslední měřený čas každé RH\nRH80: 30 s; ostatní: 600 s\n\n"
        "Podmíněná 2D rekonstrukce\nStejné barevné měřítko\nČerně: elektrody\nBíle: měřený řez",
        va="top", transform=axes.ravel()[len(cases)].transAxes, fontsize=11, linespacing=1.5)
    fig.colorbar(im, ax=used, label="Modelový KPFM signál [V]", shrink=.85, pad=.02)
    fig.suptitle("Přehled RH — 2D mapy v posledním čase", fontsize=17)
    fig.savefig(output / "prehled_mapy.png", dpi=190)
    plt.close(fig)
    fig, axes = plt.subplots(rows, columns, figsize=(14, rows*3.6), squeeze=False, layout="constrained")
    for ax, case in zip(axes.ravel(), cases):
        m = case["measurement"]
        ax.plot(m.x_um, m.voltage[:, 0], color=".55", lw=1, label="První měření")
        ax.plot(m.x_um, m.voltage[:, -1], color="#1675ac", lw=.8, alpha=.6)
        ax.plot(m.x_um[::5], m.voltage[::5, -1], ".", color="#1675ac", ms=3, label="Poslední měření")
        ax.plot(m.x_um, case["prediction"][:, -1], color="#d56513", lw=1.8, label="Model v posledním čase")
        ax.set(title=f"RH {m.rh:g} % · poslední t = {time_label(m.times_s[-1])}",
               xlabel="x [µm]", ylabel="KPFM [V]", ylim=limits)
        ax.grid(alpha=.18)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    axes.ravel()[len(cases)].legend(handles, labels, loc="upper left", frameon=False, fontsize=10)
    axes.ravel()[len(cases)].text(.02, .5, "Řez y = 5 µm\n\nPrvní a poslední čas\nVšechny časy jsou v rezy.png\nu jednotlivých RH.",
                                transform=axes.ravel()[len(cases)].transAxes, va="top", fontsize=11, linespacing=1.5)
    fig.suptitle("Přehled RH — naměřené a modelové podélné řezy", fontsize=17)
    fig.savefig(output / "prehled_rezy.png", dpi=190)
    plt.close(fig)


def save_initial_state(case, output):
    fig, ax = plt.subplots(figsize=(6.8, 5.8), layout="constrained")
    initial = case["fields"]["initial_phi_V"].copy()
    initial[~case["fields"]["dielectric_mask"]] = np.nan
    im = map_image(ax, initial, case,
                   (case["config"]["right_voltage_V"], case["config"]["left_voltage_V"]), "viridis")
    # Exact rectangular initial metal region, without an interpolation halo
    # leaking from charged metal nodes into the uncharged dielectric.
    for patch, voltage in zip(ax.patches, (case["config"]["left_voltage_V"], case["config"]["right_voltage_V"])):
        patch.set_facecolor(im.cmap(im.norm(voltage)))
    fig.colorbar(im, ax=ax, label="Modelový potenciál φ [V]")
    ax.set_title("Okamžik zapnutí · t = 0 s\nLevá elektroda 2 V; dielektrikum a pravá elektroda 0 V")
    fig.savefig(output / "pocatecni_stav.png", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"outputs/reference_all/reference_results.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/figures")
    parser.add_argument("--rh", type=float, nargs="+")
    args = parser.parse_args()
    saved = json.loads(args.results.read_text(encoding="utf-8"))
    c = saved["physics"]
    validate_config(c)
    measurements = {m.rh: m for m in load_all(args.data_dir, c)}
    results = sorted(saved["results"], key=lambda r: r["RH_percent"])
    if args.rh:
        missing = set(args.rh)-{r["RH_percent"] for r in results}
        if missing:
            parser.error(f"RH not present in saved fits: {sorted(missing)}")
        results = [r for r in results if r["RH_percent"] in args.rh]
    if not results:
        parser.error("No RH results to plot.")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    cases = []
    capacitance = EPS0*c["epsilon_r"]/(c["oxide_thickness_nm"]*1e-9)
    for result in results:
        m = measurements[result["RH_percent"]]
        print(f"Reconstructing RH{m.rh:g} ...", flush=True)
        reference = Reference(m, c, grid=result["grid"])
        delay = result.get("measurement_time_offset_s", 0.)
        fields = reference.fields(result["D_eff_m2_s"], result["gain"], result["offset_V"], time_offset_s=delay)
        initial = reference.fields(result["D_eff_m2_s"], result["gain"], result["offset_V"], times_s=[0.])
        fields["initial_phi_V"] = initial["phi_V"][0]
        fields["initial_V_KPFM"] = initial["V_KPFM"][0]
        sigma = capacitance*(fields["phi_V"]-c["backgate_voltage_V"])
        sigma[:, ~fields["dielectric_mask"]] = np.nan
        fields["sigma_eff_uC_m2"] = sigma*1e6
        folder = args.output/f"RH{m.rh:g}"
        folder.mkdir(parents=True, exist_ok=True)
        prediction = reference.predict(result["D_eff_m2_s"], result["gain"], result["offset_V"], delay)
        np.savez_compressed(folder/"mapy_data.npz", **fields)
        csv_data = np.column_stack((np.repeat(m.x_um, len(m.times_s)),
                                    np.tile(m.times_s, len(m.x_um)),
                                    np.tile(m.times_s+delay, len(m.x_um)),
                                    m.voltage.ravel(), prediction.ravel()))
        np.savetxt(folder/"rezy_data.csv", csv_data, delimiter=",", comments="",
                   header="x_um,t_s,t_sim_s,V_KPFM_measured,V_KPFM_model")
        cases.append(dict(measurement=m, config=c, result=result, fields=fields,
                          prediction=prediction, folder=folder))

    def limits(key, zero=False):
        low = min(float(np.nanmin(case["fields"][key])) for case in cases)
        high = max(float(np.nanmax(case["fields"][key])) for case in cases)
        if key == "V_KPFM":
            low = min(low, min(float(case["measurement"].voltage.min()) for case in cases))
            high = max(high, max(float(case["measurement"].voltage.max()) for case in cases))
        quantum = .1 if key != "sigma_eff_uC_m2" else 25
        return (min(0., math.floor(low/quantum)*quantum) if zero else math.floor(low/quantum)*quantum,
                max(quantum, math.ceil(high/quantum)*quantum))

    ranges = {key: limits(key) for key in ("V_KPFM", "phi_V", "sigma_eff_uC_m2")}
    for case in cases:
        print(f"Drawing RH{case['measurement'].rh:g} ...", flush=True)
        save_maps(case, "V_KPFM", ranges["V_KPFM"], "Modelový KPFM signál [V]",
                  "modelový KPFM signál ve všech časech", "mapy_kpfm.png")
        save_maps(case, "phi_V", ranges["phi_V"], "Modelový potenciál φ [V]",
                  "kalibrovaný potenciál ve všech časech", "mapy_potencial.png")
        save_maps(case, "sigma_eff_uC_m2", ranges["sigma_eff_uC_m2"], "σ_eff [µC/m²]",
                  "efektivní kapacitní náboj; SiO₂ 300 nm", "mapy_naboj.png", "cividis")
        save_cuts(case, ranges["V_KPFM"])
    save_overviews(cases, args.output, ranges["V_KPFM"])
    if c["initial_condition"] == "left_electrode_only":
        save_initial_state(cases[0], args.output)
    metadata = {"source_results": str(args.results.resolve()), "physics": c,
                "results": results, "color_ranges": ranges,
                "map_source": "independent_numerical_transport_model_not_PINN_output",
                "orientation": "arrays: time,y,x; plotted origin lower",
                "notes": [f"Initial condition: {c['initial_condition']}.",
                          "With left_electrode_only the first measured profile is not used as the initial 2D state.",
                          "A measurement time offset is fitted within the configured 0-3 second limit.",
                          "The illustrative RH60 image is not used.",
                          "D at search bound is unresolved, not a measured coefficient.",
                          "sigma_eff is a local-capacitance approximation, not uniquely reconstructed trapped charge."]}
    (args.output/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output/"popis.txt").write_text(
        "2D mapy a řezy pro všechny vybrané RH\n\n"
        "prehled_mapy.png: všechny RH v posledním měřeném čase; RH80 = 30 s, ostatní = 600 s.\n"
        "prehled_rezy.png: porovnání prvního a posledního měření s modelem.\n"
        "RH*/mapy_kpfm.png: modelový KPFM signál v každém čase měření.\n"
        "RH*/mapy_potencial.png: kalibrovaný modelový potenciál.\n"
        "RH*/mapy_naboj.png: efektivní kapacitní náboj, SiO₂ 300 nm, kov je maskovaný.\n"
        "RH*/rezy.png: podélný měřený/modelový řez, příčný modelový řez a odchylky.\n"
        "NPZ a CSV obsahují čísla použitá v obrázcích.\n\n"
        "Mapy jsou výstup numerického referenčního modelu, nikoli PINN nebo měřené 2D mapy.\n"
        f"Počáteční podmínka: {c['initial_condition']}. Viz pocatecni_stav.png.\n"
        "Dielektrikum začíná bez náboje v rámci lokálního kapacitního modelu.\n"
        "Čas měření 0 je od zapnutí posunut o fitovanou prodlevu nejvýše 3 s.\n"
        "Neshoda s prvními řezy může ukazovat chybějící okamžité elektrostatické pole v transportním modelu.\n"
        "Je-li D na mezi hledání, není rozlišené — viz metadata.json.\n"
        "Pro stejnou veličinu je barevná stupnice společná všem RH a časům.\n",
        encoding="utf-8")
    archive = args.output.parent/f"{args.output.name}_obrazky.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(args.output))
    print(f"Saved {len(cases)} RH sets in {args.output}\nArchive: {archive}", flush=True)


if __name__ == "__main__":
    main()
