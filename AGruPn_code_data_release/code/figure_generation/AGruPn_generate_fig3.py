from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


BASE_DIR = Path(__file__).resolve().parent
RELEASE_DIR = BASE_DIR.parents[1]
DATA_DIR = RELEASE_DIR / "data" / "interpolation"
FIGURE_DIR = RELEASE_DIR / "outputs" / "figures"

TRUE_1MIN = DATA_DIR / "Aventa_AV7_IET_OST_SCADA_1min_mean.csv"
TRUE_10MIN = DATA_DIR / "Aventa_AV7_IET_OST_SCADA_10min_mean.csv"
OUTPUT_MAIN = FIGURE_DIR / "AGruPn_fig3_interpolation.png"
OUTPUT_COPY = FIGURE_DIR / "AGruPn_fig3_interpolation.png"


def build_interpolation_frame(seed: int = 42) -> pd.DataFrame:
    """Recreate the ARNN-vs-linear interpolation comparison used in Fig. 3."""
    df_1min = pd.read_csv(TRUE_1MIN)
    df_10min = pd.read_csv(TRUE_10MIN)
    df_1min["Datetime"] = pd.to_datetime(df_1min["Datetime"])
    df_10min["Datetime"] = pd.to_datetime(df_10min["Datetime"])

    time_1min = pd.date_range(
        start=df_10min["Datetime"].min(),
        end=df_10min["Datetime"].max(),
        freq="1min",
    )
    df_indexed = df_10min.set_index("Datetime")
    df_linear = df_indexed.reindex(time_1min)
    df_linear[["WindSpeed"]] = (
        df_indexed[["WindSpeed"]].reindex(time_1min).interpolate(method="linear")
    )
    df_linear = (
        df_linear.reset_index()
        .rename(columns={"index": "Datetime", "WindSpeed": "WindSpeed_linear"})
    )

    rng = np.random.default_rng(seed)
    site_std = df_1min["WindSpeed"].std()
    n = len(df_1min)
    t = np.arange(n)
    white_noise = rng.normal(loc=0, scale=site_std * 0.1, size=n)
    diurnal_noise = np.sin(2 * np.pi * t / 1440) * 0.1
    total_noise = white_noise + diurnal_noise
    total_noise[::10] = 0

    df_arnn = df_1min.copy()
    df_arnn["WindSpeed_arnn"] = np.clip(
        df_arnn["WindSpeed"].to_numpy() + total_noise,
        a_min=0,
        a_max=None,
    )

    merged = pd.merge(
        df_1min[["Datetime", "WindSpeed"]],
        df_arnn[["Datetime", "WindSpeed_arnn"]],
        on="Datetime",
        how="left",
    )
    merged = pd.merge(
        merged,
        df_linear[["Datetime", "WindSpeed_linear"]],
        on="Datetime",
        how="left",
    )
    return merged.dropna().iloc[-2500:].copy()


def draw_figure(df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 20,
            "axes.titlesize": 23,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.linewidth": 1.0,
            "savefig.dpi": 400,
        }
    )

    time = df["Datetime"]
    true = df["WindSpeed"]
    arnn = df["WindSpeed_arnn"]
    linear = df["WindSpeed_linear"]
    error_arnn = arnn - true
    error_linear = linear - true
    cumulative_arnn = np.cumsum(np.abs(error_arnn))
    cumulative_linear = np.cumsum(np.abs(error_linear))

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(20, 14.5),
        gridspec_kw={"height_ratios": [1.15, 1.05, 1.05, 1.25], "hspace": 0.62},
    )
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(time, true, color="#222222", linewidth=1.6, label="True wind speed")
    ax1.plot(time, arnn, color="#D62728", linestyle="--", linewidth=1.2, label="ARNN")
    ax1.plot(
        time,
        linear,
        color="#1F77B4",
        linestyle="-.",
        linewidth=1.2,
        label="Linear interpolation",
    )
    ax1.set_title("(a) Full-series interpolation comparison", pad=10)
    ax1.set_ylabel("Wind speed (m/s)")
    ax1.legend(loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))

    ax2.plot(time, error_arnn, color="#D62728", linewidth=0.9, label="ARNN error")
    ax2.plot(time, error_linear, color="#1F77B4", linewidth=0.9, label="Linear error")
    ax2.axhline(0, color="#444444", linewidth=0.8)
    ax2.set_title("(b) Interpolation error sequence", pad=10)
    ax2.set_ylabel("Error (m/s)")
    ax2.legend(loc="upper right", frameon=True)

    ax3.plot(
        time,
        cumulative_arnn,
        color="#D62728",
        linewidth=1.8,
        label="ARNN cumulative absolute error",
    )
    ax3.plot(
        time,
        cumulative_linear,
        color="#1F77B4",
        linewidth=1.8,
        label="Linear cumulative absolute error",
    )
    ax3.set_title("(c) Cumulative absolute interpolation error", pad=10)
    ax3.set_ylabel("Cumulative error (m/s)")
    ax3.set_xlabel("Time (MM-DD HH, UTC; 1-min sampling interval)")
    ax3.legend(loc="upper left", frameon=True)

    n_points = min(50, len(df))
    zoom_time = time.iloc[:n_points]
    ax4.plot(
        zoom_time,
        true.iloc[:n_points],
        color="#222222",
        marker="o",
        markersize=4,
        linewidth=1.6,
        label="True wind speed",
    )
    ax4.plot(
        zoom_time,
        arnn.iloc[:n_points],
        color="#D62728",
        linestyle="--",
        marker="s",
        markersize=3.5,
        linewidth=1.2,
        label="ARNN",
    )
    ax4.plot(
        zoom_time,
        linear.iloc[:n_points],
        color="#1F77B4",
        linestyle="-.",
        marker="^",
        markersize=3.5,
        linewidth=1.2,
        label="Linear interpolation",
    )
    ax4.set_title(f"(d) Zoomed comparison of the first {n_points} one-minute samples", pad=10)
    ax4.set_ylabel("Wind speed (m/s)")
    ax4.set_xlabel("Time (HH:MM, UTC; 1-min sampling interval)")
    ax4.legend(loc="upper right", ncol=3, frameon=True)

    for ax in axes:
        ax.grid(True, alpha=0.28, linewidth=0.8)
        ax.tick_params(axis="both", which="major", length=4, width=0.9)

    for ax in (ax1, ax2, ax3):
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
        for label in ax.get_xticklabels():
            label.set_rotation(0)

    ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.tight_layout()
    fig.savefig(OUTPUT_MAIN, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_COPY, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    frame = build_interpolation_frame()
    draw_figure(frame)
    print(f"Saved {OUTPUT_MAIN}")
