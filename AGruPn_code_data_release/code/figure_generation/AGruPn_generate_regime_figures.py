from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


CODE_DIR = Path(__file__).resolve().parent
RELEASE_DIR = CODE_DIR.parents[1]
FIGURE_DIR = RELEASE_DIR / "outputs" / "figures"
REGIME_DIR = RELEASE_DIR / "data" / "regime_evaluation"
VOTE_DIR = RELEASE_DIR / "data" / "vote_consistency"


COLORS = {
    "blue": "#376CB0",
    "gold": "#F0B600",
    "orange": "#D9822B",
    "red": "#C44536",
    "green": "#3C8D75",
    "dark": "#2E2E2E",
    "grid": "#D9D9D9",
}


def set_style(base: int = 24) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": base,
            "axes.labelsize": base + 3,
            "axes.titlesize": base + 4,
            "xtick.labelsize": base,
            "ytick.labelsize": base,
            "legend.fontsize": base - 2,
            "axes.linewidth": 1.6,
            "lines.linewidth": 3.0,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_both(fig: plt.Figure, filename: str, subdir: Path | None = None) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_param_kappa() -> None:
    set_style(24)
    df = pd.read_csv(REGIME_DIR / "bollinger_parameter_sweep.csv")
    sub = df[df["window"] == 120].copy()
    pivot = sub.pivot(index="beta", columns="k", values="kappa").sort_index().sort_index(axis=1)

    cmap = LinearSegmentedColormap.from_list(
        "kappa_map", ["#FFF7EC", "#FDD49E", "#FC8D59", "#D7301F", "#7F0000"]
    )
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
    ax.set_xlabel(r"Band width coefficient $\alpha$")
    ax.set_ylabel(r"Volatility multiplier $\beta$")
    ax.set_title(r"Cohen's kappa sensitivity ($L=120$ min)", pad=14, fontweight="bold")

    threshold = (np.nanmin(pivot.values) + np.nanmax(pivot.values)) / 2
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                fontsize=22,
                color="white" if v > threshold else COLORS["dark"],
                fontweight="bold",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Kappa", rotation=270, labelpad=28)
    cbar.ax.tick_params(labelsize=22)
    save_both(fig, "AGruPn_param_kappa.png", CODE_DIR)


def plot_param_pr() -> None:
    set_style(24)
    df = pd.read_csv(REGIME_DIR / "bollinger_parameter_sweep.csv")
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    marker_map = {120: "o", 240: "s", 360: "^", 500: "D", 720: "P"}
    color_map = {120: COLORS["blue"], 240: "#5DA5DA", 360: COLORS["green"], 500: COLORS["orange"], 720: COLORS["red"]}

    for window in sorted(df["window"].unique()):
        sub = df[df["window"] == window]
        sizes = 110 + 260 * sub["f1"] / max(sub["f1"].max(), 1e-9)
        ax.scatter(
            sub["recall"],
            sub["precision"],
            s=sizes,
            marker=marker_map.get(int(window), "o"),
            color=color_map.get(int(window), COLORS["dark"]),
            alpha=0.74,
            edgecolors="white",
            linewidths=1.2,
            label=f"L={int(window)}",
        )

    best = df.sort_values(["f1", "kappa"], ascending=False).head(3)
    for _, row in best.iterrows():
        ax.annotate(
            rf"$L$={int(row['window'])}, $\alpha$={row['k']:.1f}, $\beta$={row['beta']:.1f}",
            (row["recall"], row["precision"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=19,
            color=COLORS["dark"],
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall trade-off", pad=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=1.0)
    ax.legend(ncol=2, frameon=True, framealpha=0.92)
    save_both(fig, "AGruPn_param_precision_recall.png", CODE_DIR)


def load_vote_data() -> pd.DataFrame:
    df = pd.read_csv(VOTE_DIR / "ten_min_vote_comparison.csv")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df.set_index("Datetime")


def plot_10min_timeline(compare: pd.DataFrame) -> None:
    set_style(25)
    density = compare["label_truth_10min"].rolling(72, min_periods=12).mean()
    center = density.idxmax()
    segment = compare.loc[center - pd.Timedelta(hours=12) : center + pd.Timedelta(hours=12)].copy()
    if segment.empty:
        segment = compare.iloc[:300].copy()

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    ax.plot(segment.index, segment["wind_10min"], color=COLORS["blue"], lw=3.2, label="10-min wind speed")
    ymin = segment["wind_10min"].min() - 0.35
    ymax = segment["wind_10min"].max() + 0.35
    ax.fill_between(
        segment.index,
        ymin,
        ymax,
        where=segment["label_truth_10min"].eq(1),
        step="pre",
        alpha=0.24,
        color=COLORS["gold"],
        label="Rule label",
    )
    ax.fill_between(
        segment.index,
        ymin,
        ymax,
        where=segment["label_pred_10min"].eq(1),
        step="pre",
        alpha=0.18,
        color=COLORS["red"],
        label="Bollinger label",
    )
    ax.set_ylabel("Wind speed (m/s)")
    ax.set_xlabel("Time (MM-DD HH:MM, UTC)")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=1.0)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)
    save_both(fig, "AGruPn_10min_regime_timeline.png", CODE_DIR)


def plot_10min_cm() -> None:
    set_style(26)
    raw = pd.read_csv(VOTE_DIR / "ten_min_vote_confusion_matrix.csv")
    cm = raw[["Pred_Normal", "Pred_Gusty"]].to_numpy()
    cmap = LinearSegmentedColormap.from_list("cm_blue", ["#F7FBFF", "#DEEBF7", "#9ECAE1", "#3182BD", "#08519C"])
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Gusty"])
    ax.set_yticklabels(["Normal", "Gusty"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Reference label")
    ax.set_title("10-min label confusion matrix", pad=14, fontweight="bold")
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=28,
                fontweight="bold",
                color="white" if cm[i, j] > threshold else COLORS["dark"],
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=28)
    cbar.ax.tick_params(labelsize=22)
    save_both(fig, "AGruPn_10min_confusion_matrix.png", CODE_DIR)


def plot_10min_ratio(compare: pd.DataFrame) -> None:
    set_style(25)
    sample = compare.sample(min(12000, len(compare)), random_state=42)
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.scatter(
        sample["truth_ratio"],
        sample["pred_ratio"],
        s=36,
        color=COLORS["blue"],
        alpha=0.38,
        edgecolors="none",
    )
    ax.plot([0, 1], [0, 1], "--", color=COLORS["dark"], lw=2.6, label=r"$y=x$")
    ax.axvline(0.5, ls="--", color=COLORS["orange"], lw=2.6, label="Reference threshold")
    ax.axhline(0.5, ls=":", color=COLORS["red"], lw=3.0, label="Predicted threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Reference gust ratio")
    ax.set_ylabel("Predicted gust ratio")
    ax.set_title("10-min voting-ratio agreement", pad=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=1.0)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)
    save_both(fig, "AGruPn_10min_voting_ratio.png", CODE_DIR)


def plot_10min_cases(compare: pd.DataFrame) -> None:
    set_style(26)
    truth = compare["label_truth_10min"].to_numpy()
    pred = compare["label_pred_10min"].to_numpy()
    cases = np.where(
        (truth == 0) & (pred == 0),
        "TN",
        np.where((truth == 0) & (pred == 1), "FP", np.where((truth == 1) & (pred == 0), "FN", "TP")),
    )
    counts = pd.Series(cases).value_counts().reindex(["TN", "FP", "FN", "TP"]).fillna(0)
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    bars = ax.bar(counts.index, counts.values, color=[COLORS["blue"], COLORS["orange"], COLORS["red"], COLORS["green"]], edgecolor="white", linewidth=1.3)
    ax.set_ylabel("Count")
    ax.set_xlabel("Case type")
    ax.set_title("Distribution of 10-min label cases", pad=14, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25, linewidth=1.0)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value):,}", ha="center", va="bottom", fontsize=24)
    save_both(fig, "AGruPn_10min_case_distribution.png", CODE_DIR)


def grouped_metric_figure(data: dict, order: list[str], filename: str, panel_start: str) -> None:
    set_style(24)
    metrics = ["SMAPE", "ICR", "IS"]
    units = {"SMAPE": "SMAPE (%)", "ICR": "ICR (%)", "IS": "IS"}
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.2), gridspec_kw={"wspace": 0.28})
    x = np.arange(len(order))
    width = 0.36
    legend_handles = None
    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        global_vals = [data["Global Model"][name][metric] for name in order]
        regional_vals = [data["Regional Model"][name][metric] for name in order]
        bars_g = ax.bar(x - width / 2, global_vals, width, label="Global", color=COLORS["blue"], alpha=0.86)
        bars_r = ax.bar(x + width / 2, regional_vals, width, label="Regional", color=COLORS["gold"], alpha=0.92)
        if legend_handles is None:
            legend_handles = [bars_g[0], bars_r[0]]
        ax.set_title(f"({chr(ord(panel_start) + idx)}) {units[metric]}", pad=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=18, ha="right")
        ax.set_ylabel(units[metric])
        ymax = max(max(global_vals), max(regional_vals))
        ax.set_ylim(0, ymax * 1.18)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25, linewidth=1.0)
        ax.tick_params(length=5, width=1.4)
        lower_is_better = metric in {"SMAPE", "IS"}
        for pos, gv, rv in zip(x, global_vals, regional_vals):
            improvement = (gv - rv) / gv * 100 if lower_is_better else (rv - gv) / gv * 100
            label_y = max(gv, rv) + ymax * 0.018
            ax.text(
                pos,
                label_y,
                f"{improvement:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=18,
                color=COLORS["dark"],
                bbox=dict(facecolor="white", edgecolor="#CCCCCC", alpha=0.82, pad=2),
                clip_on=False,
            )
    fig.legend(
        legend_handles,
        ["Global", "Regional"],
        loc="upper center",
        ncol=2,
        frameon=True,
        framealpha=0.92,
        bbox_to_anchor=(0.5, 1.04),
    )
    save_both(fig, filename, None)


def plot_fig6() -> None:
    data = {
        "Global Model": {
            "Site1": {"SMAPE": 3.576, "ICR": 90.035, "IS": 3.800},
            "Site2": {"SMAPE": 3.422, "ICR": 91.084, "IS": 4.300},
            "Site3": {"SMAPE": 3.278, "ICR": 89.336, "IS": 4.101},
            "Aventa": {"SMAPE": 13.076, "ICR": 86.492, "IS": 3.244},
            "MDGT": {"SMAPE": 5.321, "ICR": 84.394, "IS": 5.466},
        },
        "Regional Model": {
            "Site1": {"SMAPE": 2.709, "ICR": 90.737, "IS": 3.600},
            "Site2": {"SMAPE": 2.709, "ICR": 91.263, "IS": 3.800},
            "Site3": {"SMAPE": 2.789, "ICR": 88.815, "IS": 3.612},
            "Aventa": {"SMAPE": 11.849, "ICR": 87.103, "IS": 2.839},
            "MDGT": {"SMAPE": 4.552, "ICR": 85.305, "IS": 5.310},
        },
    }
    grouped_metric_figure(data, ["Site1", "Site2", "Site3", "Aventa", "MDGT"], "AGruPn_fig8_metric_summary.pdf", "i")


def plot_fig7() -> None:
    data = {
        "Global Model": {
            "Spring": {"SMAPE": 4.524, "ICR": 84.740, "IS": 5.903},
            "Summer": {"SMAPE": 6.407, "ICR": 81.847, "IS": 6.524},
            "Autumn": {"SMAPE": 4.307, "ICR": 83.353, "IS": 5.353},
            "Winter": {"SMAPE": 6.047, "ICR": 87.634, "IS": 4.082},
        },
        "Regional Model": {
            "Spring": {"SMAPE": 3.945, "ICR": 86.247, "IS": 5.772},
            "Summer": {"SMAPE": 5.778, "ICR": 81.807, "IS": 6.312},
            "Autumn": {"SMAPE": 3.875, "ICR": 84.225, "IS": 5.174},
            "Winter": {"SMAPE": 4.609, "ICR": 88.942, "IS": 3.980},
        },
    }
    grouped_metric_figure(data, ["Spring", "Summer", "Autumn", "Winter"], "AGruPn_fig9_seasonal_metrics.pdf", "l")


def main() -> None:
    plot_param_kappa()
    plot_param_pr()
    compare = load_vote_data()
    plot_10min_timeline(compare)
    plot_10min_cm()
    plot_10min_ratio(compare)
    plot_10min_cases(compare)
    print("Regenerated parameter and 10-min consistency figures. Fig. 8 and Fig. 9 can be restored by AGruPn_restore_metric_figures.py.")


if __name__ == "__main__":
    main()
