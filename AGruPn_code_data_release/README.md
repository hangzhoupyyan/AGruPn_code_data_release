# AGruPn Code and Data Release

This repository contains the public code and data released for the revised AGruPn wind-speed forecasting study.

## Scope of This Release

The release is limited to code and data. It includes public preprocessing, regime-recognition, metric-computation, figure-generation, and partial training/notebook implementations. The complete integrated AGruPn core model, some unpublished symbolic-physics modules, raw private datasets, trained checkpoints, and manuscript files are not included because the authors are extending these components in follow-up research.

## Directory Layout

- `code/training/`: public training protocol and interfaces.
- `code/notebooks/`: partial implementation notebooks used during the study.
- `code/metrics/`: metric and regime-switching comparison scripts.
- `code/figure_generation/`: scripts that regenerate public analysis figures from the released CSV files.
- `data/interpolation/`: one-minute and ten-minute wind-speed interpolation data.
- `data/prediction_results/`: released prediction sequences for all, normal, and gust regimes.
- `data/regime_evaluation/`: Bollinger-regime parameter and consistency data.
- `data/vote_consistency/`: ten-minute voting-consistency data.

Generated outputs are written to `outputs/` and are intentionally excluded from version control.

## Quick Start

```bash
pip install -r requirements.txt
python code/training/AGruPn_public_training_pipeline.py
python code/metrics/AGruPn_compute_condition_prediction_metrics.py
python code/metrics/AGruPn_run_regime_switching_comparison.py
python code/figure_generation/AGruPn_generate_fig3.py
python code/figure_generation/AGruPn_generate_fig4_fig5.py
python code/figure_generation/AGruPn_generate_regime_figures.py
```

XGBoost is optional. If it is not installed, the regime-switching script still runs the AGruPn and Random Forest branches.

## Notes for Reviewers

The released code is intended to improve transparency and reproducibility of the reported public analyses. The main experimental data, prediction sequences, evaluation scripts, and partial model implementations are provided. The final integrated AGruPn core is omitted from this public package to protect ongoing follow-up research, but the data protocol, evaluation procedure, and public interfaces are documented in `code/training/AGruPn_public_training_pipeline.py`.
