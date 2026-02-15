# Pygame Simulation & Dashboard — Where and How

Quick reference for the live visualization and command center included in the GITHUB bundle.

---

## Pygame Simulation

- **Code**: `GITHUB/code/simulation/pygame_simulation.py` (same as `d2tl/simulation/pygame_simulation.py`).
- **Run** (from repo root):
  - Local mode: `python3 d2tl/simulation/pygame_simulation.py`
  - Live API: `python3 d2tl/simulation/pygame_simulation.py --api` (requires Selector 8000, MLP 8001, Mamba 8002).
- **Features**: 18 RSUs on map, vehicle path, weather/density zones, real-time Power/QoS/Trigger, MLP (blue) vs Mamba (red) decisions.
- **Controls**: SPACE pause, R reset, 1–4 weather, 0 auto, ESC quit.

---

## Dashboard (Streamlit)

- **Code**: `GITHUB/code/dashboard/app.py` and `GITHUB/code/dashboard/.streamlit/config.toml` (same as `d2tl/dashboard/`).
- **Run** (from repo root): `streamlit run d2tl/dashboard/app.py --server.port 8501` (APIs must be running).
- **One-command start** (APIs + Dashboard): `bash d2tl/start_all_services.sh` (script in `GITHUB/code/start_all_services.sh`).
- **Features**: Mission Control, coverage map, live prediction, scenario playbook, alerts, event log, RSU roster, system health, experiments, training & physics, reports, settings.

---

## Dependencies

- Pygame: `pip install pygame`
- Dashboard: `pip install streamlit` (and httpx, plotly, pandas, etc. as in main README).

All paths above assume the repository root is `Model_3_Coverage_Mamba3`.
