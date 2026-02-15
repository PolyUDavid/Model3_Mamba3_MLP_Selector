# Architecture — D²TL Dual-Path Design (First-Person)

I describe here the system and model architecture I designed and implemented for physics-aware 6G RSU coverage prediction.

---

## System Overview

I use a **microservice orchestrator** pattern: one brain (Selector) and two model services (MLP and Mamba). The Selector always calls both services; it uses the MLP output for normal conditions and switches to the Mamba output when my physics analyzer flags extreme scenarios (long distance, heavy rain, dense urban, or large MLP–Mamba divergence).

- **Selector Brain (Port 8000)** — My orchestrator. It runs a rule-based PhysicsAnalyzer over the request (distance, weather, density, interference), computes a trigger score, and chooses MLP or Mamba. I keep decision history and stats for transparency.
- **MLP Service (Port 8001)** — My primary path. I use an 8-layer MLP (CoverageMLP) with LayerNorm and GELU, 13 inputs and 5 outputs (Power, SINR, Radius, Area, QoS). I optimized it for latency (~0.5 ms) and accuracy (R²≈0.934).
- **Mamba Service (Port 8002)** — My physics backup. I use CoverageMamba3 (selective state-space model, 8 blocks, d_model=256). It learns distance–power decay and rain attenuation (6.97–9.73 dB). It runs concurrently so it can be selected when the Selector activates it.

---

## CoverageMLP (Primary Backbone)

I implemented an 8-layer feedforward network:

- **Input**: 13 features (RSU positions, tx power, antenna, distance, angle, building density, weather, vehicle density, interferers, rx height, frequency).
- **Backbone**: Linear(13→256) → LayerNorm → GELU, then 7 × [Linear(256→256) → LayerNorm → GELU → Dropout(0.1)].
- **Heads**: Five separate Linear(256→1) for received_power_dbm, sinr_db, coverage_radius_m, coverage_area_km2, qos_score.
- **Init**: Xavier (gain=0.5) for stable training.

I use the same 13-D input and 5-D output as Mamba so the Selector can compare the two fairly. Code: `d2tl/mlp_service/model.py` and `paper_package/04_model_code/mlp_backbone.py`. Diagram: `paper_package/08_backbones/plots/MLP_backbone_architecture.png`.

---

## CoverageMamba3 (Physics Backup Backbone)

I implemented a Mamba-style SSM for coverage so that the model can capture temporal/physical structure (path-loss decay, rain accumulation):

- **Input projection**: Linear(13→256) → LayerNorm → GELU → Dropout(0.1). I treat a single vector as sequence length 1.
- **Blocks**: 8 × MambaBlock. Each block: LayerNorm → Linear(256→512) split into (x, gate) → SelectiveSSM(512) → SiLU(gate)*x → Linear(512→256) + residual.
- **SelectiveSSM**: I use Conv1d(k=4), then project to Δ, B, C; I discretize A with clamping for stability; I implement a stable SSM step (gated cumulative style) and skip connection D.
- **Output**: LayerNorm → mean over sequence → five Linear(256→1) heads, same as MLP.

I tuned d_model=256, d_state=16, d_conv=4, expand=2. Code: `models/mamba3_coverage.py` and `paper_package/04_model_code/mamba3_backbone.py`. Diagram: `paper_package/08_backbones/plots/Mamba3_backbone_architecture.png`. Detailed spec: `paper_package/08_backbones/BACKBONE_MAMBA3.md`.

---

## Selector Brain Logic

I designed the decision logic so that Mamba is used only when it adds value (e.g. long distance where it beats MLP in my Exp4):

- **Primary trigger**: Long distance (>500 m, with higher score for >700 m).
- **Compound**: Distance + heavy rain, or distance + urban density, or range + high interference.
- **Triple compound**: I add extra score when multiple risk factors are present.
- **Divergence**: If |MLP_power − Mamba_power| > threshold, I can override to Mamba when trigger_score ≥ 0.15.

I cap the trigger at 1.0 and use a threshold (e.g. 0.3) to decide “use Mamba”. All of this is implemented in `d2tl/selector_brain/selector.py`.

---

## Data Flow (End-to-End)

1. Client sends a coverage request (13-D or API fields) to the Selector (8000).
2. Selector runs PhysicsAnalyzer and calls MLP (8001) and Mamba (8002) in parallel.
3. Selector compares outputs and divergence; it returns either MLP or Mamba result, plus metadata (trigger_score, reasons, both predictions).
4. I log decisions and activation rate for analysis and dashboard.

This architecture lets me keep MLP speed for the majority of traffic while still using Mamba’s physics when it matters.
