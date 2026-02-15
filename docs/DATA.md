# Data — What I Use and Where I Put It (First-Person)

I describe the datasets and JSON files I use for training, validation, and experiments, and where I store them so the repo is self-contained and reproducible.

---

## Coverage Dataset (13-D Input, 5-D Output)

I use a single coverage dataset aligned with 6G RSU / V2I scenarios. Each sample has:

- **Inputs (13)**: rsu_x_position_m, rsu_y_position_m, tx_power_dbm, antenna_tilt_deg, antenna_azimuth_deg, distance_to_rx_m, angle_to_rx_deg, building_density, weather_condition, vehicle_density_per_km2, num_interferers, rx_height_m, frequency_ghz.
- **Targets (5)**: received_power_dbm, sinr_db, coverage_radius_m, coverage_area_km2, qos_score.

I normalize inputs (and optionally targets) using per-feature mean/std; I save these as `feature_stats` in the checkpoint so that APIs and experiments use the same normalization.

---

## Where I Keep the Data

**Full train/val/test and splits**

I keep the canonical split and full-size JSON in one place so I don’t duplicate large files:

- **Folder**: `Model_3_Complete_Package/training_data_package/`
- **Files**: train_data.json, val_data.json, test_data.json, split_indices.json, coverage_training_data_v2.json (and any data explanation docs there).
- **Index**: I list all paths and alternate locations (e.g. Model_3_V2_FINAL, Model_3_FINAL, training_data/) in **`paper_package/01_training_data/MANIFEST.txt`**.

**Training history (for plots)**

I copy the training-history JSON (epochs, train_loss, val_loss, val_mae) into `paper_package/01_training_data/` so one script can plot every run:

- mlp_training_history.json  
- mlp_service_training_history.json  
- mamba_training_history.json  
- training_history_v2_final.json  
- training_history_training_dir.json  

---

## Experiment Data (JSON)

I write all experiment outputs to JSON so that tables and figures can be regenerated without re-running experiments (unless desired):

- **D²TL 7 experiments**: `paper_package/02_experiment_data/all_experiment_results.json` (exp1–exp7). Original is also under `d2tl/experiments/results/all_experiment_results.json`.
- **Temporal / sequence**: `paper_package/02_experiment_data/temporal_results.json`, `temporal_experiment_results.json`.
- **First-principles validation**: `paper_package/02_experiment_data/first_principles_validation.json`.

I describe each file in **`paper_package/02_experiment_data/MANIFEST.txt`**.

---

## Data Flow in My Pipeline

1. **Training**: I load train/val (and optionally test) from the paths above; I compute feature_stats and save them in the checkpoint. I log training history to JSON.
2. **APIs**: I load the checkpoint (and feature_stats) so that raw API input is normalized the same way as in training.
3. **Experiments**: I load the same checkpoints and, when needed, the same test set or synthetic evaluation set; I write results to the experiment JSON files.
4. **Visualizations**: I read training-history JSON from `paper_package/01_training_data/` and experiment JSON from `paper_package/02_experiment_data/` and generate all plots under `paper_package/07_visualizations/plots/`.

I do not ship the full train/val/test JSON inside the GITHUB folder; I reference them via MANIFEST and the paths above so that the GITHUB folder stays a complete “map” of the repo while large data stays in one canonical place. Anyone cloning the repo can place or generate data there and follow the same layout.

---

## Summary Table

| What I provide        | Where it lives |
|-----------------------|----------------|
| Train/val/test paths  | paper_package/01_training_data/MANIFEST.txt → Model_3_Complete_Package/training_data_package/ (and alternatives) |
| Training history JSON | paper_package/01_training_data/*.json |
| Experiment JSON       | paper_package/02_experiment_data/*.json |
| Plot script           | paper_package/07_visualizations/plot_all_training_and_experiments.py |
| Full file index       | paper_package/INDEX.md |

This way, the model, code, experiments, and data are all covered and findable from the GITHUB folder and the paper_package index.
