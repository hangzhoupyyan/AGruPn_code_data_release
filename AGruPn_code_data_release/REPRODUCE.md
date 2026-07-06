# Reproduction Guide

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Inspect the public training protocol.

```bash
python code/training/AGruPn_public_training_pipeline.py
```

3. Recompute condition-wise prediction metrics.

```bash
python code/metrics/AGruPn_compute_condition_prediction_metrics.py
```

4. Reproduce regime-switching comparison outputs.

```bash
python code/metrics/AGruPn_run_regime_switching_comparison.py
```

5. Regenerate analysis figures from the released CSV files.

```bash
python code/figure_generation/AGruPn_generate_fig3.py
python code/figure_generation/AGruPn_generate_fig4_fig5.py
python code/figure_generation/AGruPn_generate_regime_figures.py
python code/figure_generation/AGruPn_restore_metric_figures.py
```

All generated files are saved under `outputs/`.
