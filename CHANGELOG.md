# Changelog

All notable updates to the D²TL (Model 3) release are listed here. I maintain this for transparency and reproducibility.

---

## [1.1] — 2026-02 (GITHUB bundle for Git)

### Added / Updated

- **GITHUB/code/simulation/** — Pygame simulation script (18 RSU, local/API) + README.
- **GITHUB/code/dashboard/** — Streamlit app + .streamlit config + README.
- **GITHUB/code/experiments/** — run_all_experiments.py, run_all_experiments_via_api.py, validate_experiment_data.py, EXPERIMENT_REPORT.md + README.
- **GITHUB/code/start_all_services.sh** — One-command start for MLP, Mamba, Selector, Dashboard.
- **GITHUB/data/experiment/all_experiment_results_via_api.json** — Seven experiments via real API (physical-unit MSE, latency).
- **GITHUB/data/figures/backbones/** — MLP, Mamba3, D2TL full architecture PNGs.
- **GITHUB/data/figures/visualizations/** — Training loss/MAE and Exp1–Exp7 + temporal plots (01–09).
- **GITHUB/code/** — Backbone diagram scripts (generate_mlp_backbone_diagram.py, generate_mamba_backbone_diagram.py, generate_d2tl_full_architecture.py).
- **Docs**: README, REPO_STRUCTURE, data/README, data/experiment/MANIFEST, code/README updated for Pygame, Dashboard, experiments, and figures.

---

## [1.0] — 2026-02

### Released

- **D²TL system**: Selector Brain (8000) + MLP Service (8001) + Mamba Service (8002).
- **Backbones**: CoverageMLP (8-layer, 256 hidden, ~469K params) and CoverageMamba3 (8-block SSM, ~13.7M params).
- **Training**: MLP and Mamba training scripts; training-history JSON; canonical train/val/test paths documented.
- **Seven experiments**: Scenario distribution, distance–power decay, rain coverage, stratified MSE, latency/params, ablation, tail risk. Full JSON results and plots.
- **Dashboard**: Streamlit Smart Coverage Command Center (Mission Control, coverage map, live prediction, scenario playbook, alerts, event log, RSU roster, system health, experiments, training & physics, reports, settings).
- **APIs**: FastAPI for Selector, MLP, Mamba with health, metrics, physics report, and raw predict for orchestration.
- **Paper package**: Consolidated training data index, experiment data, model code copies, API/dashboard READMEs, visualization script and plots, backbone docs and architecture diagrams.
- **GITHUB folder**: First-person README, ARCHITECTURE, EXPERIMENTS, TRAINING, API_AND_SERVICES, DATA docs, CITATION, LICENSE, REPO_STRUCTURE, CHANGELOG.

### Notes

- Checkpoints (best_mlp_coverage.pth, best_coverage.pth) are produced by training; paths are documented in paper_package/03_model_data/MANIFEST.txt.
- All experiment and training visualizations can be regenerated with paper_package/07_visualizations/plot_all_training_and_experiments.py.
