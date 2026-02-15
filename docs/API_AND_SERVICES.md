# API and Services — What I Expose (First-Person)

I expose three HTTP services: the Selector Brain (orchestrator) and the two model services (MLP and Mamba). Here I summarize what I implemented and how to run them.

---

## Ports and Roles

| Service        | Port | Role |
|----------------|------|------|
| Selector Brain | 8000 | Orchestrator: calls both MLP and Mamba, returns one result + metadata |
| MLP Service    | 8001 | Primary path: CoverageMLP inference |
| Mamba Service  | 8002 | Physics backup: CoverageMamba3 inference |

I start all three (e.g. with `d2tl/start_all_services.sh`) so that the dashboard and any client can talk to the Selector at 8000; the Selector calls 8001 and 8002 internally.

---

## Selector Brain (8000)

I implement the “brain” that decides when to use MLP vs Mamba.

- **POST /predict** — Body: same coverage input fields (rsu_x_position_m, tx_power_dbm, distance_to_rx_m, weather_condition, building_density, etc.). I return the chosen prediction (MLP or Mamba), selected_model, trigger_score, risk_level, reasons, both mlp_prediction and mamba_prediction, divergence info, and latencies.
- **POST /predict/batch** — Same for a list of samples; I return a list of results plus mamba_activations count.
- **GET /health** — Status of Selector and both downstream services (MLP, Mamba) and aggregate stats (total_requests, mlp_decisions, mamba_decisions, mamba_activation_rate).
- **GET /stats** — Decision statistics (requests, activation rate, recent activations).
- **GET /decisions/history** — Recent decision history (optional limit).
- **GET /architecture** — Text description of the dual-path architecture and trigger conditions.

Code: `d2tl/selector_brain/selector.py`.

---

## MLP Service (8001)

I implement the primary-path API so the Selector (and optionally external clients) can get fast MLP predictions.

- **POST /predict** — JSON body with coverage input fields. I return received_power_dbm, sinr_db, coverage_radius_m, coverage_area_km2, qos_score, inference_time_ms, model="MLP".
- **POST /predict/batch** — List of samples; I return a list of predictions and count.
- **POST /predict/raw** — For Selector: body is a list of 13 normalized features. I return the same outputs plus raw_output and inference_time_ms.
- **GET /health** — status, model, parameters, checkpoint_loaded, device.
- **GET /metrics** — model, parameters, val_metrics, checkpoint path.

Code: `d2tl/mlp_service/api.py`. Model: `d2tl/mlp_service/model.py`. Checkpoint: `d2tl/mlp_service/best_mlp_coverage.pth` (or path in script).

---

## Mamba Service (8002)

I implement the physics-backup API so the Selector can get Mamba predictions and optional physics notes.

- **POST /predict** — Same input shape as MLP. I return the same five outputs plus inference_time_ms, model="Mamba-3", and optional physics_note (e.g. rain attenuation, long-range, dense urban). I also append to an internal data buffer for observation.
- **POST /predict/batch** — List of samples; same as above.
- **POST /predict/raw** — 13-D normalized features for Selector; returns outputs and raw_output.
- **POST /activate** — Called by Selector when Mamba is chosen; I log an activation event (reason, timestamp).
- **GET /health** — status, model, parameters, checkpoint_loaded, best_r2, data_buffer_size, activation_count.
- **GET /metrics** — model, parameters, best_r2, val_metrics, physics_capabilities, data_buffer_size, activation_count.
- **GET /physics** — I return a short “physics report”: path-loss slope errors, rain attenuation range (6.97–9.73 dB), density scaling, and a short text on why Mamba helps for physics.
- **GET /data/buffer** — Status and recent entries of the observation buffer (for debugging or dashboard).

Code: `d2tl/mamba_service/api.py`. Model: `models/mamba3_coverage.py`. Checkpoint: `training/best_coverage.pth` (parent dir of d2tl = Model_3_Coverage_Mamba3).

---

## How I Run the Services

From the repo root (Model_3_Coverage_Mamba3):

```bash
bash d2tl/start_all_services.sh
```

Or manually (three terminals):

```bash
# Terminal 1 — MLP
cd d2tl/mlp_service && uvicorn api:app --host 0.0.0.0 --port 8001

# Terminal 2 — Mamba
cd d2tl/mamba_service && uvicorn api:app --host 0.0.0.0 --port 8002

# Terminal 3 — Selector
cd d2tl/selector_brain && uvicorn selector:app --host 0.0.0.0 --port 8000
```

The dashboard (Streamlit) then talks to the Selector (8000) for live prediction and status; the Selector calls 8001 and 8002. I designed it this way so that a single entry point (8000) is enough for clients and the dashboard, while I keep the two backbones as separate services for clarity and independent scaling.
