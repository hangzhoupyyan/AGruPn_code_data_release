from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


BASE_DIR = Path(__file__).resolve().parent
RELEASE_DIR = BASE_DIR.parents[1]
DATA_DIR = RELEASE_DIR / "data" / "prediction_results"
FIGURE_DIR = RELEASE_DIR / "outputs" / "figures"


def load_pair(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load global and regime-specific prediction results."""
    global_df = pd.read_csv(DATA_DIR / f"{prefix}_all.csv")
    regional_parts = [
        pd.read_csv(DATA_DIR / f"{prefix}_0.csv"),
        pd.read_csv(DATA_DIR / f"{prefix}_1.csv"),
    ]
    regional_df = (
        pd.concat(regional_parts, ignore_index=True)
        .sort_values("time_index")
        .reset_index(drop=True)
    )
    return prepare(global_df), prepare(regional_df)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["true_value"] >= 0.1].copy()
    df["error"] = df["pred_med"] - df["true_value"]
    df["abs_error"] = df["error"].abs()
    df["rel_error"] = (df["abs_error"] / df["true_value"]) * 100
    return df


def fit_line(df: pd.DataFrame) -> tuple[float, float]:
    return tuple(np.polyfit(df["true_value"], df["pred_med"], 1))


def draw_pair_grid(items: list[tuple[str, str]], panel_start: str, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 29,
            "axes.labelsize": 34,
            "axes.titlesize": 42,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "legend.fontsize": 25,
            "axes.linewidth": 1.8,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    n_cols = len(items)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(34, 14.8),
        gridspec_kw={"height_ratios": [1.0, 1.02], "wspace": 0.30, "hspace": 0.34},
    )

    colors = {
        "global": "#4C72B0",
        "regional": "#F2C94C",
        "diag": "#2F2F2F",
    }
    panel_ord = [chr(ord(panel_start) + i) for i in range(n_cols)]

    for col, ((name, prefix), panel) in enumerate(zip(items, panel_ord)):
        global_df, regional_df = load_pair(prefix)
        ax = axes[0, col]

        ax.scatter(
            global_df["true_value"],
            global_df["pred_med"],
            s=16,
            color=colors["global"],
            alpha=0.42,
            edgecolors="none",
            label="Global",
        )
        ax.scatter(
            regional_df["true_value"],
            regional_df["pred_med"],
            s=16,
            color=colors["regional"],
            alpha=0.50,
            edgecolors="none",
            label="Regional",
        )

        min_val = min(
            global_df["true_value"].min(),
            global_df["pred_med"].min(),
            regional_df["true_value"].min(),
            regional_df["pred_med"].min(),
        )
        max_val = max(
            global_df["true_value"].max(),
            global_df["pred_med"].max(),
            regional_df["true_value"].max(),
            regional_df["pred_med"].max(),
        )
        x_range = np.linspace(min_val, max_val, 100)
        ax.plot(x_range, x_range, "--", color=colors["diag"], lw=2.8, label="y=x")

        g_a, g_b = fit_line(global_df)
        r_a, r_b = fit_line(regional_df)
        ax.plot(x_range, g_a * x_range + g_b, color=colors["global"], lw=3.0)
        ax.plot(x_range, r_a * x_range + r_b, color="#F5B700", lw=3.0)

        ax.set_title(f"({panel}) {name}", pad=12, fontweight="bold")
        ax.set_xlabel("Measured (m/s)")
        ax.set_ylabel("Predicted (m/s)" if col == 0 else "")
        ax.grid(True, alpha=0.26, linestyle="--", linewidth=1.0)
        ax.tick_params(length=6, width=1.5)
        ax.legend(loc="lower right", frameon=True, framealpha=0.88, borderpad=0.35)

        text = (
            f"Global: y={g_a:.3f}x+{g_b:.3f}\n"
            f"Regional: y={r_a:.3f}x+{r_b:.3f}"
        )
        ax.text(
            0.04,
            0.94,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=25,
            bbox=dict(facecolor="white", edgecolor="#BBBBBB", alpha=0.86, pad=4),
        )

        ax_h = axes[1, col]
        max_error = min(
            100,
            max(global_df["rel_error"].quantile(0.995), regional_df["rel_error"].quantile(0.995)),
        )
        bins = np.linspace(0, max_error, 36)
        ax_h.hist(
            global_df["rel_error"],
            bins=bins,
            density=True,
            alpha=0.50,
            color=colors["global"],
            edgecolor="white",
            linewidth=0.8,
            label="Global",
        )
        ax_h.hist(
            regional_df["rel_error"],
            bins=bins,
            density=True,
            alpha=0.58,
            color=colors["regional"],
            edgecolor="white",
            linewidth=0.8,
            label="Regional",
        )
        g_mean = global_df["rel_error"].mean()
        r_mean = regional_df["rel_error"].mean()
        ax_h.axvline(g_mean, color=colors["global"], linestyle="--", lw=3.0)
        ax_h.axvline(r_mean, color="#F5B700", linestyle="--", lw=3.0)
        ax_h.set_xlabel("Relative error (%)")
        ax_h.set_ylabel("Density" if col == 0 else "")
        ax_h.grid(True, alpha=0.26, linestyle="--", linewidth=1.0)
        ax_h.tick_params(length=6, width=1.5)

        ax_h.text(
            0.04,
            0.94,
            f"Mean RE\nG={g_mean:.1f}%, R={r_mean:.1f}%",
            transform=ax_h.transAxes,
            va="top",
            ha="left",
            fontsize=22,
            bbox=dict(facecolor="white", edgecolor="#BBBBBB", alpha=0.86, pad=4),
        )

        inset = inset_axes(ax_h, width="44%", height="42%", loc="center right", borderpad=1.4)
        for data, color, marker in [
            (global_df["abs_error"].to_numpy(), colors["global"], "o"),
            (regional_df["abs_error"].to_numpy(), "#F5B700", "s"),
        ]:
            sorted_err = np.sort(data)
            cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
            limit = np.percentile(sorted_err, 95)
            mask = sorted_err <= limit
            inset.plot(sorted_err[mask], cdf[mask], color=color, lw=2.2)
            for q in [0.2, 0.5, 0.8]:
                idx = np.searchsorted(cdf, q)
                inset.plot(sorted_err[idx], cdf[idx], marker, color=color, ms=5)
        inset.set_title("CDF of abs. error", fontsize=18, pad=4)
        inset.set_xlabel("Abs. error (m/s)", fontsize=15)
        inset.set_ylabel("CDF", fontsize=15)
        inset.tick_params(axis="both", labelsize=14, length=3)
        inset.grid(True, alpha=0.20)

    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    seasonal_items = [
        ("Spring", "quantile_prediction_results_T1spring_9"),
        ("Summer", "quantile_prediction_results_T1summer_9"),
        ("Autumn", "quantile_prediction_results_T1autumn_9"),
        ("Winter", "quantile_prediction_results_T1winter_9"),
    ]
    dataset_items = [
        ("Aventa", "quantile_prediction_results_Aventa_9"),
        ("Site1", "quantile_prediction_results_penglai1_9"),
        ("Site2", "quantile_prediction_results_penglai2_9"),
        ("Site3", "quantile_prediction_results_penglai3_9"),
    ]
    draw_pair_grid(seasonal_items, "a", FIGURE_DIR / "AGruPn_fig4_seasonal_comparison.pdf")
    draw_pair_grid(dataset_items, "e", FIGURE_DIR / "AGruPn_fig5_dataset_comparison.pdf")
    print("Saved AGruPn_fig4_seasonal_comparison.pdf and AGruPn_fig5_dataset_comparison.pdf")
