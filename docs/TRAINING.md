# Training — How I Train Both Backbones (First-Person)

I describe how I train the MLP (primary) and Mamba-3 (physics backup) and where I store data and checkpoints.

---

## Data I Use

I use a single coverage dataset with 13 input features and 5 targets (received_power_dbm, sinr_db, coverage_radius_m, coverage_area_km2, qos_score). I normalize inputs (and optionally targets) using feature stats saved in the checkpoint.

- **Canonical splits**: I keep train/val/test and split indices in `Model_3_Complete_Package/training_data_package/` (train_data.json, val_data.json, test_data.json, split_indices.json, coverage_training_data_v2.json). Paths are listed in `paper_package/01_training_data/MANIFEST.txt`.
- **Training history JSON**: I save epoch-wise train_loss, val_loss, and val_mae for every run. Copies used for plotting are in `paper_package/01_training_data/` (e.g. mlp_training_history.json, mamba_training_history.json, training_history_v2_final.json, training_history_training_dir.json).

---

## Training the MLP (Primary Path)

I train CoverageMLP with:

- **Script**: `d2tl/training/train_mlp.py`
- **Model**: `d2tl/mlp_service/model.py` (CoverageMLP, 13→256×8→5)
- **Output**: Checkpoint and feature stats written to `d2tl/mlp_service/best_mlp_coverage.pth` (or path configured in the script). Training history is saved (e.g. `d2tl/training/mlp_training_history.json` / `d2tl/mlp_service/training_history.json`).

I use standard MSE (or multi-task loss) on the 5 targets, validation MAE for early stopping, and the same normalization as in the Mamba pipeline when both are trained from the same data.

---

## Training the Mamba-3 (Physics Backup)

I train CoverageMamba3 with:

- **Script**: `training/train_coverage.py` (or the version under `Model_3_Complete_Package/source_code/`). I use the same 13-D input and 5-D output and compatible normalization.
- **Model**: `models/mamba3_coverage.py` (CoverageMamba3)
- **Output**: Checkpoint (e.g. `training/best_coverage.pth`) with model_state_dict, optional feature_stats, epoch, r2, val_metrics. Training history is saved under `training/training_history.json` or the path set in the script.

I train on the same (or compatible) data so that MLP and Mamba see the same feature distribution; the Selector then chooses based on physics rules and divergence, not on a train-time split of “who gets which scenario.”

---

## Reproducing My Training Curves

I provide multiple training-history JSON files so that all reported curves can be reproduced or inspected:

- **MLP (D²TL)**: `paper_package/01_training_data/mlp_training_history.json`
- **MLP service**: `paper_package/01_training_data/mlp_service_training_history.json`
- **Mamba-3**: `paper_package/01_training_data/mamba_training_history.json`
- **Mamba V2 final**: `paper_package/01_training_data/training_history_v2_final.json`
- **Training dir**: `paper_package/01_training_data/training_history_training_dir.json`

To plot all training loss and val MAE in one figure:

```bash
python paper_package/07_visualizations/plot_all_training_and_experiments.py
```

Output: `paper_package/07_visualizations/plots/01_training_loss_and_val_mae.png`.

---

## Checkpoints and Model Data

I do not commit large `.pth` files to the repo by default. I document where I expect them to be so that services and experiments can load them:

- **MLP**: `d2tl/mlp_service/best_mlp_coverage.pth` (see `paper_package/03_model_data/MANIFEST.txt`)
- **Mamba**: `training/best_coverage.pth`

Metadata (e.g. model_metadata.json, checkpoint_info.txt) is under `Model_3_Complete_Package/trained_model/`. After training, place the checkpoint in the path above (or set the path in the API/experiment scripts) so that the dashboard, APIs, and `run_all_experiments.py` run without change.

---

## Summary

I train the MLP for low latency and high R² on the full data; I train the Mamba-3 for physics consistency (path-loss, rain attenuation) on the same or compatible data. I save training histories and checkpoint paths so that anyone can reproduce my curves and run the full pipeline (APIs, experiments, dashboard) from the same checkpoints and data layout described here.
