# Repository Structure — Where Everything Lives (First-Person)

I organize the repo so that the **GITHUB** folder is the entry point for documentation and citation, and **paper_package** is the single place for data, code copies, and visualizations. The rest of the repo contains the live code and data paths. This file maps “what I provide” to “where it is.”

---

## GITHUB/ (this folder)

- **README.md** — First-person overview, quick start, results summary, repo structure table.
- **CITATION.bib** / **CITATION.md** — How to cite my work.
- **LICENSE** — MIT.
- **REPO_STRUCTURE.md** — This file.
- **code/** — **Full code bundle (model + API + data generator + training + simulation + dashboard + experiments)**:
  - **mlp_service/** — model.py (CoverageMLP), api.py (FastAPI 8001).
  - **mamba_service/** — api.py (FastAPI 8002).
  - **selector_brain/** — selector.py (FastAPI 8000).
  - **models/** — mamba3_coverage.py (CoverageMamba3).
  - **data_generator/** — generate_coverage_data_v2.py, generate_coverage_data.py.
  - **training/** — train_mlp.py, train_coverage.py.
  - **simulation/** — pygame_simulation.py (18 RSU, local/API), README.md.
  - **dashboard/** — app.py (Streamlit), .streamlit/config.toml, README.md.
  - **experiments/** — run_all_experiments.py, run_all_experiments_via_api.py, validate_experiment_data.py, EXPERIMENT_REPORT.md, README.md.
  - **start_all_services.sh** — Start MLP, Mamba, Selector, Dashboard.
  - **generate_*_backbone*.py** — Diagram scripts (MLP, Mamba3, D2TL full).
  - **README.md** — How to run and path notes.
- **data/** — **All experiment and training JSON + figures**:
  - **experiment/** — all_experiment_results.json, all_experiment_results_via_api.json, temporal_*.json, first_principles_validation.json, MANIFEST.txt.
  - **training/** — mlp_training_history.json, mamba_training_history.json, training_history_*.json, MANIFEST.txt.
  - **figures/backbones/** — MLP_backbone_architecture.png, Mamba3_backbone_architecture.png, D2TL_full_architecture.png.
  - **figures/visualizations/** — 01_training_loss_and_val_mae.png … 09_temporal_results.png (training + Exp1–Exp7 + temporal).
  - **README.md** — Description of each file and figures.
- **docs/** — Detailed first-person docs:
  - **ARCHITECTURE.md** — System and dual-path design; MLP and Mamba-3 backbones; Selector logic.
  - **EXPERIMENTS.md** — All seven experiments and where to find data/plots.
  - **TRAINING.md** — How I train both models; data and checkpoint paths; training history.
  - **API_AND_SERVICES.md** — Ports, endpoints, and how to run the three services.
  - **DATA.md** — Dataset description; where train/val/test and experiment JSON live.

---

## paper_package/

I put all “paper-ready” assets here so one index covers everything.

- **README.md** — Short package overview and folder list.
- **INDEX.md** — File-level index (training data, experiment data, model code, API, dashboard, visualizations, backbones).
- **01_training_data/** — Training-history JSON (copies) + MANIFEST.txt with paths to full train/val/test.
- **02_experiment_data/** — all_experiment_results.json, temporal_*.json, first_principles_validation.json + MANIFEST.txt.
- **03_model_data/** — MANIFEST.txt for checkpoint paths and metadata.
- **04_model_code/** — mlp_backbone.py, mamba3_backbone.py + README.
- **05_api_code/** — README with run instructions and endpoints (live code in d2tl/).
- **06_dashboard/** — README for running the Streamlit dashboard (live app in d2tl/dashboard/).
- **07_visualizations/** — plot_all_training_and_experiments.py + plots/ (training loss, val MAE, Exp1–Exp7, temporal).
- **08_backbones/** — BACKBONE_MLP.md, BACKBONE_MAMBA3.md; diagram scripts; plots/ (MLP and Mamba-3 architecture PNGs).

---

## d2tl/

Live implementation of the dual-path system and dashboard.

- **mlp_service/** — model.py (CoverageMLP), api.py (FastAPI 8001), best_mlp_coverage.pth (after training).
- **mamba_service/** — api.py (FastAPI 8002); loads models/mamba3_coverage.py and training/best_coverage.pth.
- **selector_brain/** — selector.py (FastAPI 8000), PhysicsAnalyzer, decision logic.
- **training/** — train_mlp.py; writes MLP checkpoint and training history.
- **experiments/** — run_all_experiments.py, EXPERIMENT_REPORT.md; results/all_experiment_results.json, results/plots/.
- **dashboard/** — app.py (Streamlit), .streamlit/config.toml.
- **start_all_services.sh** — Start MLP, Mamba, Selector.

---

## models/

- **mamba3_coverage.py** — CoverageMamba3 (SelectiveSSM, MambaBlock, multi-task heads). Used by d2tl/mamba_service and d2tl/experiments.

---

## training/

- **train_coverage.py** — Mamba-3 training script.
- **best_coverage.pth** — Mamba checkpoint (after training).
- **training_history.json** — Mamba training curves (copy in paper_package/01_training_data/).

---

## Data locations (canonical)

- **Model_3_Complete_Package/training_data_package/** — train_data.json, val_data.json, test_data.json, split_indices.json, coverage_training_data_v2.json (and docs).
- **Model_3_Complete_Package/trained_model/** — model_metadata.json, checkpoint_info.txt (and any packaged checkpoint references).
- **validation_results/** — first_principles_validation.json (copy in paper_package/02_experiment_data/).
- **temporal_experiment/results/** — temporal_results.json, temporal_experiment_results.json (copies in paper_package/02_experiment_data/).

---

## One-line summary

- **Read first**: GITHUB/README.md and GITHUB/docs/.
- **Find any file**: paper_package/INDEX.md.
- **Run services**: bash d2tl/start_all_services.sh (MLP, Mamba, Selector, Dashboard).
- **Run Pygame**: python3 d2tl/simulation/pygame_simulation.py or ... --api.
- **Run experiments**: d2tl/experiments/run_all_experiments.py or run_all_experiments_via_api.py.
- **Experiment & figure data**: GITHUB/data/experiment/ (JSON), GITHUB/data/figures/ (backbones + visualizations).
- **Regenerate plots**: paper_package/07_visualizations/plot_all_training_and_experiments.py; paper_package/08_backbones/generate_*.py.
- **Cite**: GITHUB/CITATION.bib and GITHUB/CITATION.md.

I keep the GITHUB folder and paper_package together so that the repo is complete and self-explanatory for reproduction and citation.
