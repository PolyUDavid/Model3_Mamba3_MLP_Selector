# D²TL: Physics-Aware Dual-Path RSU Coverage Prediction for 6G V2I

I propose and implement **D²TL** (Dual-Path Transfer Learning): a physics-aware dual-path system for 6G RSU coverage and signal-strength prediction. In this repository I release the full model suite—two backbone architectures (MLP primary + Mamba-3 physics backup), a Selector Brain orchestrator, training pipelines, seven experiments, a Streamlit command-center dashboard, and all data and visualizations needed to reproduce and extend the work.

---

## What I Built

- **Two backbones**: (1) **CoverageMLP** — 8-layer MLP, ~469K params, R²≈0.934, ~0.5 ms inference; (2) **CoverageMamba3** — selective state-space model, ~13.7M params, learns Friis path-loss and ITU-R rain attenuation (6.97–9.73 dB).
- **Selector Brain** (Port 8000): I implement an orchestrator that always calls both MLP (8001) and Mamba (8002), and uses Mamba’s output when extreme physics are detected (long distance, heavy rain, dense urban, or model divergence).
- **Seven experiments**: scenario distribution, distance–power decay, rain coverage collapse, stratified performance, latency/params, ablation, and tail-risk analysis—all with JSON results and plots.
- **Training**: I provide MLP and Mamba training scripts, training-history JSON, and canonical train/val/test splits.
- **Dashboard**: A Streamlit “Smart Coverage Command Center” with Mission Control, coverage map, live prediction, scenario playbook, alerts, event log, RSU roster, system health, experiments, training & physics, reports, and settings.
- **Pygame simulation**: Real-time vehicle-on-map visualization with 18 RSUs; local or live API mode; shows MLP vs Mamba decisions, trigger score, and conditions.
- **APIs**: FastAPI services for MLP, Mamba, and Selector with health, metrics, and physics reports.
- **Experiment data**: Both local-run and via-API experiment JSON (all_experiment_results.json, all_experiment_results_via_api.json) and all training/experiment figures in **GITHUB/data/**.

Everything is organized so you can run, reproduce, and cite the work from a single place.

---

## Quick Start

**1. Environment (Python 3.9+)**

```bash
pip install torch numpy pandas scikit-learn matplotlib streamlit folium fastapi uvicorn httpx pydantic
```

**2. Train MLP (primary path)**

```bash
python d2tl/training/train_mlp.py
# Checkpoint: d2tl/mlp_service/best_mlp_coverage.pth
```

**3. Train Mamba-3 (physics backup)** — uses existing data under `Model_3_Complete_Package/training_data_package/` or `training_data/`

```bash
python training/train_coverage.py
# Checkpoint: training/best_coverage.pth
```

**4. Run all seven experiments (no API required)**

```bash
python d2tl/experiments/run_all_experiments.py
# Writes: d2tl/experiments/results/all_experiment_results.json + plots
```

**5. Start APIs and dashboard**

```bash
bash d2tl/start_all_services.sh
# Starts MLP (8001), Mamba (8002), Selector (8000), Dashboard (8501)
```

**6. Run Pygame simulation (optional)**

```bash
# With APIs running: live API mode
python3 d2tl/simulation/pygame_simulation.py --api
# Local mode (no API): python3 d2tl/simulation/pygame_simulation.py
```

**7. Regenerate all training and experiment visualizations**

```bash
python paper_package/07_visualizations/plot_all_training_and_experiments.py
# Output: paper_package/07_visualizations/plots/
```

---

## Repository Structure (Where to Find What)

I keep everything under this repo so that model, data, code, and docs are in one place. **Inside this GITHUB folder** I also bundle the full code and data packages so the repo is self-contained.

| I provide… | Where it lives |
|------------|----------------|
| **Model code** (MLP + Mamba backbones) | **`GITHUB/code/mlp_service/model.py`**, **`GITHUB/code/models/mamba3_coverage.py`** — also `paper_package/04_model_code/`, `d2tl/mlp_service/model.py`, `models/mamba3_coverage.py` |
| **API code** (MLP, Mamba, Selector) | **`GITHUB/code/mlp_service/api.py`**, **`GITHUB/code/mamba_service/api.py`**, **`GITHUB/code/selector_brain/selector.py`** — also `d2tl/` (see `GITHUB/code/README.md`) |
| **Data generator** (training data) | **`GITHUB/code/data_generator/generate_coverage_data_v2.py`**, **`generate_coverage_data.py`** |
| **Training scripts** (MLP + Mamba) | **`GITHUB/code/training/train_mlp.py`**, **`GITHUB/code/training/train_coverage.py`** — also `d2tl/training/`, `training/` |
| **All experiment JSON** (7 experiments + temporal + validation) | **`GITHUB/data/experiment/`** — `all_experiment_results.json`, `temporal_results.json`, `temporal_experiment_results.json`, `first_principles_validation.json`, MANIFEST.txt |
| **All training JSON** (training history) | **`GITHUB/data/training/`** — `mlp_training_history.json`, `mamba_training_history.json`, `training_history_v2_final.json`, etc. + MANIFEST.txt (full train/val/test paths) |
| **Training data** (splits + history) | `paper_package/01_training_data/` (history); full train/val/test paths in `01_training_data/MANIFEST.txt` and **`GITHUB/data/training/MANIFEST.txt`** |
| **Checkpoints / model data** | Paths and metadata in `paper_package/03_model_data/MANIFEST.txt` |
| **Dashboard** | **`GITHUB/code/dashboard/app.py`** — run: `streamlit run d2tl/dashboard/app.py` |
| **Pygame simulation** | **`GITHUB/code/simulation/pygame_simulation.py`** — run: `python3 d2tl/simulation/pygame_simulation.py` or `... --api` |
| **Experiment scripts & via-API results** | **`GITHUB/code/experiments/`** (run_all_experiments.py, run_all_experiments_via_api.py); **`GITHUB/data/experiment/`** (all_experiment_results.json, all_experiment_results_via_api.json) |
| **Training & val curves + experiment plots** | **`GITHUB/data/figures/visualizations/`** (01–09 PNGs); regenerate: `paper_package/07_visualizations/plot_all_training_and_experiments.py` |
| **Backbone diagrams** | **`GITHUB/data/figures/backbones/`** (MLP, Mamba3, D2TL full PNGs); `paper_package/08_backbones/` (scripts + docs) |
| **Detailed docs** (architecture, experiments, training, API, data) | **`GITHUB/docs/`** |

- **GITHUB/code/** — Complete copies: model base code, APIs, data generator, training scripts (see `GITHUB/code/README.md`).
- **GITHUB/data/experiment/** — All experiment JSON; **GITHUB/data/training/** — All training JSON (see `GITHUB/data/README.md`).

A full file-level index is in **`paper_package/INDEX.md`**.

---

## Main Results (My Experiments)

- **Exp1** — Scenario distribution: 30K samples, 80.6% extreme (long distance, dense urban, heavy weather).
- **Exp2** — Distance–power decay: D²TL slope error vs theory (Rural/Urban) documented in experiment JSON and report.
- **Exp3** — Rain coverage: MLP/Mamba/D²TL vs physics across Clear → Heavy Rain; Mamba learns ~6.97–9.73 dB attenuation.
- **Exp4** — Stratified MSE: D²TL best on all regimes; largest gains on extreme_distance (+6.2%) and extreme_compound (+6.8%).
- **Exp5** — Latency: D²TL early-exit ~2.85 ms vs Mamba-only ~16.15 ms (≈5.7× speedup).
- **Exp6** — Ablation: D²TL Selector matches or beats MLP-only and clearly beats Mamba-only and random/naive strategies.
- **Exp7** — Tail risk: D²TL matches MLP tail (p95/p99/max) while adding physics backup when needed.

Detailed tables and plots are in `d2tl/experiments/EXPERIMENT_REPORT.md` and `paper_package/07_visualizations/plots/`.

---

## Citation

If you use this model, experiments, or code, I would appreciate a citation. See **`CITATION.bib`** and **`CITATION.md`** in this folder.

---

## License

I release the code and documentation under the **MIT License** (see `LICENSE`). Data and pretrained checkpoints are for research and reproduction; please respect any third-party data terms if you redistribute datasets.

---

## Author

**NOK KO** — Model design, implementation, experiments, dashboard, and documentation.  
D²TL: Physics-Aware Dual-Path RSU Coverage 6G V2I — Model 3 (Coverage Mamba-3).

For a complete, one-place overview of every file and folder, see **`paper_package/INDEX.md`** and **`GITHUB/docs/`**.
