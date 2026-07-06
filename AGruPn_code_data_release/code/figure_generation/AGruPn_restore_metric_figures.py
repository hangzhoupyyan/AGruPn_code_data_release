from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


RELEASE_DIR = Path(__file__).resolve().parents[2]
FIGURE_DIR = RELEASE_DIR / "outputs" / "figures"


COLORS = {
    "global": "#4E79A7",
    "regional": "#F28E2B",
    "improve": "#59A14F",
    "decline": "#E15759",
}


def set_original_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.0,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def metric_improvement(global_vals: np.ndarray, regional_vals: np.ndarray, metric: str) -> np.ndarray:
    if metric in {"SMAPE", "IS"}:
        return (global_vals - regional_vals) / global_vals * 100
    return (regional_vals - global_vals) / global_vals * 100


def draw_grouped_bars(ax, labels, global_vals, regional_vals, metric, panel_label, show_legend=False):
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, global_vals, width, color=COLORS["global"], label="Global Model")
    ax.bar(x + width / 2, regional_vals, width, color=COLORS["regional"], label="Regional Model")
    ax.set_title(f"({panel_label})", loc="center", fontweight="bold")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y")
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)


def draw_improvement_inset(ax, labels, improvements):
    inset = inset_axes(ax, width="45%", height="36%", loc="upper right", borderpad=0.9)
    y = np.arange(len(labels))
    bar_colors = [COLORS["improve"] if value >= 0 else COLORS["decline"] for value in improvements]
    inset.barh(y, improvements, color=bar_colors, alpha=0.88)
    inset.axvline(0, color="#333333", linewidth=0.8)
    inset.set_yticks([])
    inset.set_title("Promotion rate", fontsize=8, pad=2)
    inset.tick_params(axis="x", labelsize=7, length=2)
    for yi, value in zip(y, improvements):
        inset.text(
            value,
            yi,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=7,
        )
    span = max(abs(improvements).max(), 1)
    inset.set_xlim(-span * 1.25, span * 1.35)
    inset.grid(True, axis="x", alpha=0.2)


def restore_fig8() -> None:
    set_original_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["Site1", "Site2", "Site3", "Aventa", "MDGT"]
    data = {
        "SMAPE": (
            np.array([3.576, 3.422, 3.278, 13.076, 5.321]),
            np.array([2.709, 2.709, 2.789, 11.849, 4.552]),
        ),
        "ICR": (
            np.array([90.035, 91.084, 89.336, 86.492, 84.394]),
            np.array([90.737, 91.263, 88.815, 87.103, 85.305]),
        ),
        "IS": (
            np.array([3.800, 4.300, 4.101, 3.244, 5.466]),
            np.array([3.600, 3.800, 3.612, 2.839, 5.310]),
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), gridspec_kw={"wspace": 0.34})
    for ax, metric, panel in zip(axes, ["SMAPE", "ICR", "IS"], ["i", "j", "k"]):
        global_vals, regional_vals = data[metric]
        draw_grouped_bars(ax, labels, global_vals, regional_vals, metric, panel, show_legend=(metric == "SMAPE"))
        draw_improvement_inset(ax, labels, metric_improvement(global_vals, regional_vals, metric))
    fig.savefig(FIGURE_DIR / "AGruPn_fig8_metric_summary.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def restore_fig9() -> None:
    set_original_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["Spring", "Summer", "Autumn", "Winter"]
    data = {
        "SMAPE": (
            np.array([4.524, 6.407, 4.307, 6.047]),
            np.array([3.945, 5.778, 3.875, 4.609]),
        ),
        "ICR": (
            np.array([84.740, 81.847, 83.353, 87.634]),
            np.array([86.247, 81.807, 84.225, 88.942]),
        ),
        "IS": (
            np.array([5.903, 6.524, 5.353, 4.082]),
            np.array([5.772, 6.312, 5.174, 3.980]),
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), gridspec_kw={"wspace": 0.34})
    for ax, metric, panel in zip(axes, ["SMAPE", "ICR", "IS"], ["l", "m", "n"]):
        global_vals, regional_vals = data[metric]
        draw_grouped_bars(ax, labels, global_vals, regional_vals, metric, panel, show_legend=(metric == "SMAPE"))
    fig.savefig(FIGURE_DIR / "AGruPn_fig9_seasonal_metrics.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    restore_fig8()
    restore_fig9()
    print("Restored AGruPn_fig8_metric_summary.pdf and AGruPn_fig9_seasonal_metrics.pdf to the original-style layout.")
