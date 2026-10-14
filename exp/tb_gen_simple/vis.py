import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hls_smart_tb.models import normalize_model_name

# use bakend for rednering no GUI
plt.switch_backend("Agg")

DIR_CURRENT = Path(__file__).parent

DIR_DATA = DIR_CURRENT / "data"

DIR_FIGURES = DIR_CURRENT / "figures"

N_GRID = 400
X_PAD_FRAC = 0.05  # x-axis padding each side, as a fraction of the log10 range shown
MIN_BANDWIDTH_DEX = 0.05  # floor for the KDE bandwidth, in decades of cycles

# Categorical slots 1-5, fixed order (one color per sample index, never cycled)
SAMPLE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
POOLED_COLOR = "#52514e"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def parse_run_dir_name(name: str) -> tuple[str, str, int] | None:
    # <design>__<normalized model name>__<sample index>
    parts = name.split("__")
    if len(parts) < 3 or not parts[-1].isdigit():
        return None
    return parts[0], "__".join(parts[1:-1]), int(parts[-1])


def load_sample_latencies(
    dir_data: Path, design_name: str, model_name: str
) -> dict[int, dict]:
    """Map sample index -> {"latencies": [...], "csynth": (best, worst)} for every
    run of this design/model that finished co-simulation."""
    model_norm = normalize_model_name(model_name)
    samples: dict[int, dict] = {}
    for run_dir in sorted(dir_data.iterdir()):
        parsed = parse_run_dir_name(run_dir.name)
        if parsed is None or not run_dir.is_dir():
            continue
        design, model, sample_index = parsed
        if design != design_name or model != model_norm:
            continue
        run_data_fp = run_dir / "run_data.json"
        if not run_data_fp.exists():
            print(f"skip {run_dir.name}: still running or no run_data.json")
            continue
        run_data = json.loads(run_data_fp.read_text())
        data_cosim = run_data.get("data_cosim")
        if not run_data.get("cosim_ok") or not data_cosim:
            print(f"skip {run_dir.name}: no cosim data")
            continue
        latencies = data_cosim.get("transaction_latencies") or []
        if not latencies:
            print(f"skip {run_dir.name}: no per-call latencies")
            continue
        synth = run_data.get("data_synthesis") or {}
        samples[sample_index] = {
            "latencies": latencies,
            "csynth": (
                synth.get("latency_best_cycles"),
                synth.get("latency_worst_cycles"),
            ),
        }
    return samples


def bounded_kde_log(
    latencies: np.ndarray, bandwidth: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Plain Gaussian KDE of the latencies in log10(cycles) space, evaluated only
    between the min and max of the data (like seaborn's cut=0) and then
    renormalized so it integrates to 1 over that range (density per decade of
    cycles). No boundary correction: the data limits are not true edges of the
    latency distribution, so we only trim, we don't reflect. Returns
    (cycles, density), or None if all latencies are identical."""
    x = np.log10(latencies)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return None
    grid = np.linspace(lo, hi, N_GRID)
    z = (grid[:, None] - x[None, :]) / bandwidth
    density = np.exp(-0.5 * z**2).sum(axis=1)
    density /= np.trapezoid(density, grid)
    return 10**grid, density


def pooled_bandwidth(all_lat: np.ndarray) -> float:
    # Scott's rule on log10 latencies, with a floor so identical latencies still
    # give a visible bump. One bandwidth for every curve keeps them comparable.
    x = np.log10(all_lat)
    scott = x.std(ddof=1) * len(x) ** (-1 / 5) if len(x) > 1 else 0.0
    return max(float(scott), MIN_BANDWIDTH_DEX)


def _setup_axes(samples: dict[int, dict]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def _draw_rug(ax: plt.Axes, samples: dict[int, dict], y_unit: float) -> None:
    # one tick per kernel call, one row per sample, below the axis
    for rank, (sample_index, s) in enumerate(sorted(samples.items())):
        lat = np.asarray(s["latencies"])
        ax.plot(
            lat,
            np.full(len(lat), -y_unit * (rank + 1)),
            linestyle="none",
            marker="x",
            markersize=5,
            markeredgewidth=1.2,
            color=SAMPLE_COLORS[sample_index % len(SAMPLE_COLORS)],
            clip_on=False,
        )


def _csynth_bounds(samples: dict[int, dict]) -> list[tuple[float, str]]:
    # csynth best/worst latency, when csynth reports a bounded value
    best = {s["csynth"][0] for s in samples.values() if s["csynth"][0]}
    worst = {s["csynth"][1] for s in samples.values() if s["csynth"][1]}
    return [(v, "csynth best") for v in sorted(best)] + [
        (v, "csynth worst") for v in sorted(worst)
    ]


def _draw_csynth_lines(ax: plt.Axes, bounds: list[tuple[float, str]]) -> None:
    for value, name in bounds:
        ax.axvline(value, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.annotate(
            name,
            (value, 0.82),  # below the (two-row) legend at the top
            xycoords=("data", "axes fraction"),
            # worst-case label goes left of its line, best-case right, so both
            # stay inside the axes when the line sits at the plot edge
            xytext=(-3, -3) if "worst" in name else (3, -3),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="right" if "worst" in name else "left",
            fontsize=8,
            color=INK_MUTED,
        )


def _finish_axes(
    fig: plt.Figure,
    ax: plt.Axes,
    samples: dict[int, dict],
    title: str,
    y_max: float,
    save_fp: Path,
) -> Path:
    all_lat = np.concatenate([np.asarray(s["latencies"]) for s in samples.values()])
    lo, hi = float(all_lat.min()), float(all_lat.max())
    y_unit = 0.05 * y_max

    _draw_rug(ax, samples, y_unit)

    ax.set_xscale("log")
    # the x-range covers the data and the csynth bounds, so those lines always show
    csynth_bounds = _csynth_bounds(samples)
    x_lo = min([lo] + [v for v, _ in csynth_bounds])
    x_hi = max([hi] + [v for v, _ in csynth_bounds])
    pad_dex = X_PAD_FRAC * (np.log10(x_hi) - np.log10(x_lo))
    ax.set_xlim(x_lo * 10**-pad_dex, x_hi * 10**pad_dex)
    _draw_csynth_lines(ax, csynth_bounds)
    ax.set_xlabel("kernel call latency (clock cycles, co-sim)", color=INK_SECONDARY)
    ax.set_ylabel("density (per decade of cycles)", color=INK_SECONDARY)
    # headroom above the tallest curve for the csynth labels and the legend
    ax.set_ylim(bottom=-y_unit * (len(samples) + 1), top=y_max * 1.9)
    ax.set_yticks([t for t in ax.get_yticks() if t >= 0])
    ax.set_title(
        f"{title}\noverall spread {int(lo):,} to {int(hi):,} cycles ({hi / lo:.1f}x)",
        loc="center",
        fontsize=11,
        color=INK_PRIMARY,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    # dark baseline separating the density curves (above) from the per-call points (below)
    ax.axhline(0, color="#000000", linewidth=1.4, zorder=3)
    ax.set_axisbelow(True)
    # border on all four sides
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK_SECONDARY)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED)
    legend = ax.legend(
        frameon=True,
        facecolor="#ffffff",
        edgecolor=INK_MUTED,
        framealpha=1.0,
        fancybox=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),  # horizontally centered on the axes
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.set_zorder(5)  # above the gridlines and csynth lines

    fig.tight_layout()
    fig.savefig(save_fp, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {save_fp}")
    return save_fp


def plot_dists_single(
    design_name: str, model_name: str, dir_data: Path = DIR_DATA
) -> Path | None:
    # given a design name and model name, find all the samples for that
    # then on the same plot make the KDE of latencies for each sample (plus a
    # rug of the individual kernel calls), and save the plot to a file

    # save each figure as dists__<design_name>__<model_name>.png

    samples = load_sample_latencies(dir_data, design_name, model_name)
    if not samples:
        print(f"no finished samples for {design_name} / {model_name}")
        return None

    all_lat = np.concatenate([np.asarray(s["latencies"]) for s in samples.values()])
    bandwidth = pooled_bandwidth(all_lat)

    fig, ax = _setup_axes(samples)
    y_max = 0.0
    for sample_index, s in sorted(samples.items()):
        lat = np.asarray(s["latencies"])
        kde = bounded_kde_log(lat, bandwidth)  # bounded to this sample's own range
        x_cycles, density = kde if kde is not None else ([], [])
        if kde is not None:
            y_max = max(y_max, float(density.max()))
        ax.plot(
            x_cycles,
            density,
            linewidth=1.6,
            color=SAMPLE_COLORS[sample_index % len(SAMPLE_COLORS)],
            label=f"sample {sample_index} ({len(lat)} calls)",
        )

    return _finish_axes(
        fig,
        ax,
        samples,
        f"{design_name} - {model_name}",
        y_max or 1.0,
        DIR_FIGURES / f"dists__{design_name}__{normalize_model_name(model_name)}.png",
    )


def plot_dists_single_combined(
    design_name: str, model_name: str, dir_data: Path = DIR_DATA
) -> Path | None:
    # same as plot_dists_single but all samples for a given design/model are
    # pooled into one KDE; the rug still shows each sample's calls in its color

    # save each figure as dists_combined__<design_name>__<model_name>.png

    samples = load_sample_latencies(dir_data, design_name, model_name)
    if not samples:
        print(f"no finished samples for {design_name} / {model_name}")
        return None

    all_lat = np.concatenate([np.asarray(s["latencies"]) for s in samples.values()])
    bandwidth = pooled_bandwidth(all_lat)
    kde = bounded_kde_log(all_lat, bandwidth)  # bounded to the pooled range
    if kde is None:
        print(f"all latencies identical for {design_name} / {model_name}, skipping")
        return None
    x_cycles, density = kde

    fig, ax = _setup_axes(samples)
    ax.fill_between(x_cycles, density, color=POOLED_COLOR, alpha=0.18, linewidth=0)
    ax.plot(
        x_cycles,
        density,
        linewidth=1.8,
        color=POOLED_COLOR,
        label=f"pooled ({len(samples)} samples, {len(all_lat)} calls)",
    )
    # rug rows are labeled by sample via legend proxies
    for sample_index in sorted(samples):
        ax.plot(
            [],
            [],
            linestyle="none",
            marker="x",
            markersize=5,
            markeredgewidth=1.2,
            color=SAMPLE_COLORS[sample_index % len(SAMPLE_COLORS)],
            label=f"sample {sample_index}",
        )

    return _finish_axes(
        fig,
        ax,
        samples,
        f"{design_name} - {model_name} (combined)",
        float(density.max()),
        DIR_FIGURES
        / f"dists_combined__{design_name}__{normalize_model_name(model_name)}.png",
    )


def find_designs(dir_data: Path) -> list[str]:
    designs = set()
    for run_dir in dir_data.iterdir():
        parsed = parse_run_dir_name(run_dir.name)
        if parsed is not None and run_dir.is_dir():
            designs.add(parsed[0])
    return sorted(designs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot per-sample and pooled cosim latency KDEs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DIR_DATA,
        help="directory holding the <design>__<model>__<sample> run folders",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deepseek/deepseek-v4.1-flash"],
        help="model names as passed to run.py",
    )
    args = parser.parse_args()

    if not DIR_FIGURES.exists():
        DIR_FIGURES.mkdir(parents=True)

    designs = find_designs(args.data_dir)
    models = args.models

    for design_name in designs:
        for model_name in models:
            plot_dists_single(design_name, model_name, args.data_dir)
            plot_dists_single_combined(design_name, model_name, args.data_dir)
