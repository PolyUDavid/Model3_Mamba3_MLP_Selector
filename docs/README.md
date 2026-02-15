# Docs — First-Person Documentation

I wrote these documents so that the model, experiments, training, APIs, and data are all covered in one place.

| Document | What I cover |
|----------|------------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design (Selector + MLP + Mamba), CoverageMLP and CoverageMamba3 backbones, Selector logic, data flow |
| [EXPERIMENTS.md](EXPERIMENTS.md) | All seven experiments (what I ran, what I found), where results and plots live, how to run and regenerate |
| [TRAINING.md](TRAINING.md) | Data I use, how I train MLP and Mamba, checkpoint and training-history paths, reproducing curves |
| [API_AND_SERVICES.md](API_AND_SERVICES.md) | Ports (8000/8001/8002), endpoints for Selector, MLP, Mamba, and how to start services |
| [PYGAME_AND_DASHBOARD.md](PYGAME_AND_DASHBOARD.md) | Pygame simulation (18 RSU, local/API) and Streamlit dashboard — where they live and how to run |
| [DATA.md](DATA.md) | Dataset (13-D input, 5-D output), where train/val/test and experiment JSON live, data flow in the pipeline |

Start with the main [GITHUB/README.md](../README.md), then use this folder for details. For a file-level map of the whole repo, see **paper_package/INDEX.md** and **GITHUB/REPO_STRUCTURE.md**.
