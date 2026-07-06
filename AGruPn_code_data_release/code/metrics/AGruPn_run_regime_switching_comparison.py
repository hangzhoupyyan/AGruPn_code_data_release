from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "vote_consistency" / "ten_min_vote_comparison.csv"
OUT = ROOT / "outputs" / "regime_switching_comparison"


def load_xgb():
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError("xgboost is required for this comparison") from exc
    return XGBClassifier


def make_features(df):
    df = df.sort_values("Datetime").reset_index(drop=True).copy()
    wind = df["wind_10min"].astype(float)

    feature_cols = []
    for lag in [1, 2, 3, 6, 12]:
        col = f"wind_lag_{lag}"
        df[col] = wind.shift(lag)
        feature_cols.append(col)

    # Causal rolling features: only information available before the target
    # ten-minute decision interval is used by the supervised classifiers.
    past_wind = wind.shift(1)
    for window in [3, 6, 12, 24]:
        mean_col = f"past_roll_mean_{window}"
        std_col = f"past_roll_std_{window}"
        range_col = f"past_roll_range_{window}"
        slope_col = f"past_roll_slope_{window}"
        df[mean_col] = past_wind.rolling(window=window, min_periods=window).mean()
        df[std_col] = past_wind.rolling(window=window, min_periods=window).std()
        df[range_col] = (
            past_wind.rolling(window=window, min_periods=window).max()
            - past_wind.rolling(window=window, min_periods=window).min()
        )
        df[slope_col] = past_wind - wind.shift(window + 1)
        feature_cols.extend([mean_col, std_col, range_col, slope_col])

    df["target_current_reference"] = df["label_truth_10min"].astype(int)
    df["agru_pn_current_label"] = df["label_pred_10min"].astype(int)
    df["agru_pn_current_score"] = df["pred_ratio"].astype(float)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ["target_current_reference"]).reset_index(drop=True)
    return df, feature_cols


def metric_row(name, y_true, y_pred, y_score, mode):
    return {
        "Method": name,
        "Operational mode": mode,
        "Accuracy": accuracy_score(y_true, y_pred),
        "BAcc": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "PR-AUC": average_precision_score(y_true, y_score),
        "Positive rate": float(np.mean(y_pred)),
    }


def plot_curves(y_true, score_map):
    plt.figure(figsize=(6.2, 4.6), dpi=300)
    for name, score in score_map.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        auc = roc_auc_score(y_true, score)
        plt.plot(fpr, tpr, linewidth=2.0, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#777777")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves for ten-minute regime switching")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig_regime_switching_roc.png")
    plt.close()

    plt.figure(figsize=(6.2, 4.6), dpi=300)
    base = float(np.mean(y_true))
    for name, score in score_map.items():
        prec, rec, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        plt.plot(rec, prec, linewidth=2.0, label=f"{name} (AP={ap:.3f})")
    plt.axhline(base, linestyle="--", linewidth=1.2, color="#777777", label=f"Prevalence={base:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves for ten-minute regime switching")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig_regime_switching_pr.png")
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(IN_PATH, parse_dates=["Datetime"])
    df, feature_cols = make_features(raw)

    split = int(len(df) * 0.70)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    x_train = train[feature_cols].to_numpy()
    y_train = train["target_current_reference"].to_numpy()
    x_test = test[feature_cols].to_numpy()
    y_test = test["target_current_reference"].to_numpy()

    ag_pred = test["agru_pn_current_label"].to_numpy()
    ag_score = test["agru_pn_current_score"].to_numpy()

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    rf_score = rf.predict_proba(x_test)[:, 1]
    rf_pred = (rf_score >= 0.5).astype(int)

    xgb_score = None
    xgb_pred = None
    try:
        XGBClassifier = load_xgb()
        neg = max(int((y_train == 0).sum()), 1)
        pos = max(int((y_train == 1).sum()), 1)
        xgb = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=neg / pos,
        )
        xgb.fit(x_train, y_train)
        xgb_score = xgb.predict_proba(x_test)[:, 1]
        xgb_pred = (xgb_score >= 0.5).astype(int)
    except RuntimeError as exc:
        print(f"Skipping XGBoost branch: {exc}")

    metric_rows = [
        metric_row(
            "AGruPn cross-scale recognition",
            y_test,
            ag_pred,
            ag_score,
            "Current-regime recognition from within-interval fine-scale fluctuations",
        ),
        metric_row(
            "Random Forest",
            y_test,
            rf_pred,
            rf_score,
            "Next-regime prediction from past ten-minute wind features",
        ),
    ]
    if xgb_pred is not None and xgb_score is not None:
        metric_rows.append(
            metric_row(
                "XGBoost",
                y_test,
                xgb_pred,
                xgb_score,
                "Next-regime prediction from past ten-minute wind features",
            )
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "regime_switching_metrics.csv", index=False)

    pred_df = test[
        [
            "Datetime",
            "target_current_reference",
            "agru_pn_current_label",
            "agru_pn_current_score",
        ]
    ].copy()
    pred_df["rf_label"] = rf_pred
    pred_df["rf_score"] = rf_score
    if xgb_pred is not None and xgb_score is not None:
        pred_df["xgb_label"] = xgb_pred
        pred_df["xgb_score"] = xgb_score
    pred_df["previous_reference"] = df["target_current_reference"].shift(1).iloc[split:].to_numpy()
    pred_df["is_transition"] = (
        pred_df["target_current_reference"].astype(int)
        != pred_df["previous_reference"].fillna(pred_df["target_current_reference"]).astype(int)
    )
    pred_df.to_csv(OUT / "regime_switching_predictions.csv", index=False)

    trans = pred_df[pred_df["is_transition"]].copy()
    transition_rows = []
    if len(trans) > 0 and trans["target_current_reference"].nunique() > 1:
        transition_rows.extend(
            [
                metric_row(
                    "AGruPn cross-scale recognition",
                    trans["target_current_reference"].to_numpy(),
                    trans["agru_pn_current_label"].to_numpy(),
                    trans["agru_pn_current_score"].to_numpy(),
                    "Transition-window current-regime recognition",
                ),
                metric_row(
                    "Random Forest",
                    trans["target_current_reference"].to_numpy(),
                    trans["rf_label"].to_numpy(),
                    trans["rf_score"].to_numpy(),
                    "Transition-window next-regime prediction",
                ),
            ]
        )
        if xgb_pred is not None and xgb_score is not None:
            transition_rows.append(
                metric_row(
                    "XGBoost",
                    trans["target_current_reference"].to_numpy(),
                    trans["xgb_label"].to_numpy(),
                    trans["xgb_score"].to_numpy(),
                    "Transition-window next-regime prediction",
                )
            )
    pd.DataFrame(transition_rows).to_csv(OUT / "regime_switching_transition_metrics.csv", index=False)

    protocol = pd.DataFrame(
        [
            ["Reference label", "Rule-based ten-minute consensus reference label"],
            ["AGruPn/Bollinger input", "Fine-scale Bollinger detections inside the current ten-minute interval, aggregated by voting"],
            ["Classifier input", "Only historical ten-minute wind-speed features before the target interval"],
            ["Target", "Current ten-minute reference operating condition"],
            ["Chronological split", "First 70% for classifier training, last 30% for testing all methods"],
            ["Total evaluated samples", len(df)],
            ["Training samples", len(train)],
            ["Testing samples", len(test)],
            ["Test positive rate", float(np.mean(y_test))],
            ["Feature columns", ", ".join(feature_cols)],
        ],
        columns=["Item", "Setting"],
    )
    protocol.to_csv(OUT / "regime_switching_protocol.csv", index=False)

    score_map = {
        "AGruPn cross-scale": ag_score,
        "Random Forest": rf_score,
    }
    if xgb_score is not None:
        score_map["XGBoost"] = xgb_score
    plot_curves(y_test, score_map)

    print(metrics.round(4).to_string(index=False))
    if transition_rows:
        print("\nTransition windows only:")
        print(pd.DataFrame(transition_rows).round(4).to_string(index=False))
    print(f"Saved outputs to {OUT}")


if __name__ == "__main__":
    main()
