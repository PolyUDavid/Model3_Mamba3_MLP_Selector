# Paper vs Code/Simulation — Full Verification

**Paper:** D²TL: Physics-Aware Dual-Path Trigger Logic for Real-Time 6G RSU Coverage Prediction via Selective State-Space Model Integration  
**Codebase:** Model_3_Coverage_Mamba3 (data generator V2, MLP/Mamba/Selector as in repo)

**Two result files (choose one as canonical for the paper):**
- **`all_experiment_results.json`** — MSE in **scaled target space [0,1]**; latency ~0.5 / 16.15 / 2.85 ms. Paper’s Tables II–V and abstract numbers (0.003–0.006 MSE, 5.7×, 2.85 ms) match this run.
- **`all_experiment_results_via_api.json`** — MSE in **physical units** (dBm², etc.); latency ~1.42 / 60.8 / 10.32 ms (API overhead / different machine). Section 5–10 below use this file for “actual run” numbers; if you use `all_experiment_results.json` as canonical, take Table II/IV/V from that file and state “MSE in scaled target space” in the paper.

---

## 1. Equations (1)–(10) — Problem formulation

| Paper Eq | Content | Code / data | Status |
|----------|---------|-------------|--------|
| (1) | x ∈ ℝ¹³: x_rsu, y_rsu, P_tx, θ_tilt, φ_az, d, φ_rx, k, w, ρ_veh, N_int, h_rx, f | Same order in `generate_coverage_data_v2.py` and API `CoverageInput`: rsu_x_position_m, rsu_y_position_m, tx_power_dbm, antenna_tilt_deg, antenna_azimuth_deg, distance_to_rx_m, angle_to_rx_deg, building_density, weather_condition, vehicle_density_per_km2, num_interferers, rx_height_m, frequency_ghz | ✅ Match |
| (2) | y = [P_rx, γ, R_cov, A_cov, QoS] | Same 5 targets in generator and API | ✅ Match |
| (3) | λ = c/f_c ≈ 0.0508 m | `wavelength_m = 3e8 / (CARRIER_FREQ * 1e9)`, CARRIER_FREQ=5.9 | ✅ Match |
| (4) | L₀ = 20 log₁₀(4πd₀/λ) ≈ 47.86 dB | `L0 = 20 * np.log10(4 * np.pi * REFERENCE_DISTANCE / wavelength_m)` | ✅ Match |
| (5) | n_eff = n·η_obs(k), η_obs ∈ {1.0, 1.2, 1.5, 2.0} | `obstacle_factor = {0:1.0, 1:1.2, 2:1.5, 3:2.0}`, `PATH_LOSS_EXPONENT * obstacle_factor` | ✅ Match |
| (6) | PL(d) = L₀ + 10 n_eff log₁₀(d/d₀) + X_σ, σ_sh = 8 dB | `path_loss = L0 + 10*exponent*log10(d/d0)`, `shadowing_std=8.0`, `path_loss += np.random.normal(0, shadowing_std)`, `max(path_loss, 40.0)` | ✅ Match |
| (7) | A_weather(w) ∈ {0, 2, 5, 8} dB | `weather_attenuation = {0:0.0, 1:2.0, 2:5.0, 3:8.0}` | ✅ Match |
| (8) | P_rx = P_tx + G(Δφ) − PL(d) − A_weather(w) | `received_power_dbm = tx_power_dbm + antenna_gain_db - path_loss_db - weather_attenuation` | ✅ Match |
| (9) | G(Δφ) = G_max cos²(Δφ/(θ_bw/2)·π/2), G_max=10, θ_bw=120° | `angle_diff = abs(((angle_deg - azimuth_deg + 180) % 360) - 180)`, in-beam `gain = max_gain_dbi * np.cos(angle_diff / (beamwidth_deg/2) * np.pi/2)**2` | ✅ Match |
| (10) | γ = 10 log₁₀(S_lin/(I_lin+N_lin)) | `_calculate_sinr`: signal_linear, interference_linear, noise_linear; `10 * np.log10(signal_linear / (interference_linear + noise_linear))` | ✅ Match |

---

## 2. Section IV — Methodology equations

| Paper Eq | Content | Code | Status |
|----------|---------|------|--------|
| (12)–(13) | μ, σ, x̃_n = (x_n−μ)/σ, ε=10⁻⁸ | `train_mlp.py` / `train_coverage.py`: `feature_mean`, `feature_std`, `(features - mean) / (std + 1e-8)` | ✅ Match |
| (14)–(15) | t₀′=(P_rx+260)/230, t₁′=(γ+170)/230, t₂′=(R_cov−150)/90, t₃′=(A_cov−0.07)/0.12, t₄′=QoS/100 | Same in `CoverageDataset` in both training scripts | ✅ Match |
| (16) | z⁽⁰⁾ = W⁽⁰⁾ x̃ + b⁽⁰⁾, W⁽⁰⁾ ∈ ℝ²⁵⁶×¹³ | `CoverageMLP`: first layer `Linear(13, 256)` | ✅ Match |
| (17)–(18) | LayerNorm, GELU | `LayerNorm`, `GELU` in `model.py` | ✅ Match |
| (19) | h⁽ˡ⁺¹⁾ = Dropout(GELU(LN(W⁽ˡ⁾h⁽ˡ⁾+b⁽ˡ⁾))), p=0.1 | Layers 1–7: Linear, LayerNorm, GELU, Dropout(0.1) | ✅ Match |
| (20) | Five heads y_i = w_iᵀ h + b_i | `head_power`, `head_sinr`, etc. Linear(256, 1) | ✅ Match |
| **(21)** | **x_proj = W_in x_norm, W_in ∈ ℝ⁵¹²×²⁵⁶** | **Code:** `in_proj = Linear(d_model, d_inner*2)` = **Linear(256, 1024)** ⇒ **W_in ∈ ℝ¹⁰²⁴×²⁵⁶** (output 1024, then split to 512 each). | ❌ **Fix:** In paper change (21) to: “x_proj = W_in x_norm, W_in ∈ ℝ¹⁰²⁴×²⁵⁶ (output dimension d_inner×2 = 1024), then split into x_ssm and g each in ℝ⁵¹².” |
| (22) | x_conv = SiLU(Conv1d(x_ssm, K=4, groups=D)) | `SelectiveSSM`: Conv1d kernel 4, groups d_model; then `F.silu(x_conv)` | ✅ Match |
| (23)–(27) | Δ,B,C; Δ=clip(softplus(…), 0.001, 1); A=−exp(clamp(A_log,…)); w=σ(Δ), u=…; y_ssm = u⊙tanh(C)+x_conv⊙D | `x_proj` split to delta,B_param,C_param; `F.softplus(dt_proj(delta))`, `clamp(0.001, 1)`; `A = -exp(clamp(A_log,-10,3))`; `_stable_ssm`: sigmoid(delta), gated_x, + skip D | ✅ Match |
| (28)–(29) | y_gate = y_ssm ⊙ SiLU(g); x_next = x_res + W_out y_gate | `x = x * F.silu(gate)`, `out_proj(x)`, `return x + residual`; W_out is (d_model × d_inner) = 256×512 | ✅ Match |
| (30)–(34) | Trigger rules d>700 +0.35; 500<d≤700 +0.20; d>500∧w≥2 +0.20; w≥3 +0.10; d>500∧k≥2 +0.15; N_int≥3∧d>400 +0.10; n_risk≥3 +0.15; use_mamba=(s≥0.3); divergent; use_mamba_final | `selector.py` `PhysicsAnalyzer.analyze` and decision: same thresholds and logic | ✅ Match |
| (35)–(38) | L_i MSE; L = 0.15L₀+0.15L₁+0.30L₂+0.30L₃+0.10L₄; penalties; L_total | `multi_task_loss` in both training scripts: same weights and penalty terms | ✅ Match |
| (39) | Gradient clip g ← g·min(1, θ_max/‖g‖), θ_max=1 | `clip_grad_norm_(model.parameters(), 1.0)` | ✅ Match |
| (40) | R²_i = 1 − SS_res,i/(SS_tot,i+ε) | `compute_metrics`: SS_res, SS_tot, R2 per target | ✅ Match |
| (41) | T_ee = r·T_parallel + (1−r)·T_mlp, r=0.15 | `dual_effective = trigger_rate * max(MLP,Mamba) + (1-trigger_rate)*mlp_t`, trigger_rate=0.15 | ✅ Match |
| (42)–(44) | Cosine LR (MLP); warmup/stable/decay (Mamba) | `CosineAnnealingLR` T_max=150, eta_min=1e-5; `get_lr_schedule` warmup 10, stable 90, decay | ✅ Match |
| (45)–(49) | PL_max; log₁₀(R_cov); f_env=1/√η_obs; A_cov=π(R/1000)²; QoS components | `_calculate_coverage_radius`, `environment_factor = 1/sqrt(obstacle_factor)`, `coverage_area_km2 = π*(R/1000)²`, `_calculate_qos` same formulas | ✅ Match |

---

## 3. Experimental setup (Section V.A)

- **30,000 samples:** generator `num_samples=30000` ✅  
- **Train/val/test 21,000 / 4,500 / 4,500:** both training scripts use these sizes (or test = rest), seed 42 ✅  
- **RSU 0–2000 m, P_tx [30,36] dBm, d [10,1000] m, k uniform {0,1,2,3}, weather p=[0.6,0.2,0.15,0.05], vehicle density lognormal clipped [5,200]:** all in `_generate_single_sample` ✅  

---

## 4. Experiment 1 — Scenario distribution

**Paper text:** 80.6% extreme; weather 60% Clear, 20% Light, 15% Moderate, 5% Heavy; distance “~19% <200, 30% 200–500, 30% 500–800, 21% >800”.

**From `all_experiment_results_via_api.json` (exp1):**

- total: 30000, extreme: 24192, normal: 5808, **extreme_pct: 80.6** ✅  
- weather: Clear 17980, Light Rain 5971, Moderate Rain 4487, Heavy Rain 1562 → 59.93%, 19.90%, 14.96%, 5.21% ✅  
- distance: "<200" 5762, "200-500" 9099, "500-800" 9070, ">800" 6069 → 19.2%, 30.3%, 30.2%, 20.2% ✅  
- density: four categories, roughly uniform ✅  

**Verdict:** Paper Exp1 narrative and percentages match code/output.

---

## 5. Experiment 2 — Distance–power slope (Table I)

**Paper Table I:** Theory Rural −35, Urban −52.5; D²TL slope −34.1 (Rural), −50.4 (Urban); D²TL error 0.9 dB, 2.1 dB.

**From code/run (exp2):**

- Rural_theory_slope: **−35**, Urban_theory_slope: **−52.5** ✅  
- Rural_dual_slope: **−68.27**, Rural_dual_slope_error_dB: **−33.27**  
- Urban_dual_slope: **−187.65**, Urban_dual_slope_error_dB: **−135.15**  
- Rural_mlp_slope: **−69.97**, Rural_mamba_slope: **−269.77**  
- Urban_mlp_slope: **−190.90**, Urban_mamba_slope: **−460.23**  

So **current code/checkpoints do not give slope errors of 0.9 and 2.1 dB/decade**. The paper’s Table I is inconsistent with this run. To align paper with this codebase and run, replace Table I with the following (or re-run with checkpoints that yield theory-consistent slopes and then update the table).

**Suggested Table I (from actual run):**

| Environment   | Theory (dB/dec) | MLP Slope | Mamba Slope | D²TL Slope | D²TL Slope Error (dB/dec) |
|---------------|------------------|-----------|-------------|------------|----------------------------|
| Rural (k=0)   | −35.0            | −69.97    | −269.77     | −68.27     | −33.27                     |
| Urban (k=2)   | −52.5            | −190.90   | −460.23     | −187.65    | −135.15                    |

If you keep the paper’s “within 2.1 dB/decade” claim, you must either (a) use different checkpoints/results that actually achieve that, or (b) remove or soften that claim and report the above numbers.

---

## 6. Experiment 3 — Rainstorm

**Paper:** Mamba learns rain attenuation 6.97–9.73 dB (theory 8 dB); trigger increases with weather.

**From exp3 (via_api):**

- Rural: Clear dual=−73.74, Heavy Rain dual=−108.05 (MLP); Mamba Heavy=−91.77.  
- Suburban: Heavy Rain dual=−113.84 (MLP), Mamba=−106.82.  
- theory_atten: 0, 2, 5, 8 for Clear → Heavy.  
- trigger: 0 for Clear/Light/Moderate, 0.1 for Heavy (Rural and Suburban).  

So power drops with rain and trigger rises; “6.97–9.73 dB” is a separate (e.g. per-step attenuation) interpretation. For consistency, either derive 6.97–9.73 from the same exp3 outputs (e.g. attenuation from Clear to Heavy per model) or reference the script that computes that range.

---

## 7. Experiment 4 — Stratified MSE (Table II)

**Paper Table II:** MSE in ~0.003–0.006 range (scaled or normalized).

**Code:** `_mse_one` uses **physical units** (dBm, dB, m, km², QoS 0–100), so MSE is in physical units². Our run gives:

- normal: n=**895**, mlp_mse=**841.14**, mamba_mse=**10100.76**, dual_mse=**841.14**, mlp_improvement=**0%**  
- extreme_weather: n=**913**, mlp_mse=**745.79**, dual_mse=**737.29**, mlp_improvement=**1.14%**  
- extreme_distance: n=**2225**, mlp_mse=**367.45**, dual_mse=**327.50**, mlp_improvement=**10.87%**  
- extreme_density: n=**2242**, mlp_mse=**1050.29**, dual_mse=**1010.78**, mlp_improvement=**3.76%**  
- extreme_compound: n=**1594**, mlp_mse=**661.45**, dual_mse=**602.54**, mlp_improvement=**8.91%**  

So **paper Table II and “6.8%” improvement** can be aligned with code in one of two ways:

- **Option A (recommended):** Keep definition as in code: “MSE = mean of squared errors over the five outputs in physical units (power dBm, SINR dB, radius m, area km², QoS 0–100).” Then replace Table II with the following (and set “up to X% improvement” to **10.87%** for extreme_distance, **8.91%** for extreme_compound).

**Suggested Table II (physical-space MSE, from actual run):**

| Category          | n    | MLP MSE | Mamba MSE | D²TL MSE | Improvement vs MLP |
|-------------------|------|---------|-----------|----------|---------------------|
| Normal            | 895  | 841.14  | 10100.76  | 841.14   | 0%                  |
| Extreme Weather   | 913  | 745.79  | 5678.51   | 737.29   | 1.14%               |
| Extreme Distance  | 2225 | 367.45  | 328.64    | 327.50   | 10.87%              |
| Extreme Density   | 2242 | 1050.29 | 7392.17   | 1010.78  | 3.76%               |
| Compound (2+)     | 1594 | 661.45  | 1961.09   | 602.54   | 8.91%               |

- **Option B:** If you want to keep small numbers (0.003–0.006), define MSE in **scaled target space [0,1]** and add a separate evaluation script that computes and reports that scaled MSE; then table and text must say “MSE (scaled targets)” and the 6.8% (or 8.91%) must match the chosen definition.

---

## 8. Experiment 5 — Latency (Table III)

**Paper:** MLP 0.5 ms, Mamba 16.15 ms, D²TL early-exit 2.85 ms, speedup 5.7×.

**From exp5 (via_api):**

- MLP: **1.42 ms** (std 0.27), params **469,509**  
- Mamba-3: **60.80 ms** (std 3.91), params **13,735,685**  
- D²TL (parallel): **78.78 ms** (both run)  
- D²TL (early-exit): **10.32 ms** (trigger_rate=0.15)  
- speedup: **5.89** (Mamba / early-exit)  

So **Table III and abstract “5.7×”** depend on hardware. To match this run exactly, use:

**Suggested Table III (from actual run):**

| Configuration      | Parameters | Latency (ms) | Speedup vs Mamba |
|--------------------|------------|--------------|-------------------|
| MLP Only           | ~469K      | 1.42         | 42.8×             |
| Mamba Only         | ~13.7M     | 60.80        | 1.0×              |
| D²TL (Parallel)    | —          | 78.78        | —                 |
| D²TL (Early-Exit)  | —          | 10.32        | 5.89×             |

And in the paper add a note: “Latency measured on [your platform]; absolute values are hardware-dependent.” If you keep “2.85 ms” and “5.7×”, state that they come from a different (e.g. lighter) setup.

---

## 9. Experiment 6 — Ablation (Table IV)

**Paper:** MSE values ~0.004, D²TL Selector best.

**From exp6 (via_api), same physical-space MSE as Exp4/7:**

- MLP Only: **764.33**  
- Mamba Only: **6301.79**  
- Random 50/50: **3413.03**  
- Any-Extreme Trigger: **4460.16**  
- **D²TL Selector: 744.70** (best)  
- Soft Blend (0.5): **2541.32**  

**Suggested Table IV (physical-space MSE):**

| Strategy            | Overall MSE | Rank |
|---------------------|------------:|------|
| D²TL Selector       | 744.70      | 1    |
| MLP Only            | 764.33      | 2    |
| Soft Blend (0.5)     | 2541.32     | 3    |
| Any-Extreme Trigger  | 4460.16     | 4    |
| Random 50/50         | 3413.03     | 5    |
| Mamba Only           | 6301.79     | 6    |

Rank and “D²TL Selector best” match code; only the scale (physical vs scaled) and exact numbers need to match your chosen definition.

---

## 10. Experiment 7 — Tail risk (Table V)

**Paper:** Mean/median/p90/p95/p99/max in ~0.002–0.04 range; D²TL best; P99 improvement 10.8%, etc.

**From exp7 (via_api):**

- MLP: mean **764.33**, median **202.05**, p90 **2401.83**, p95 **3409.88**, p99 **4478.19**, max **6594.06**  
- Mamba: mean **6301.79**, median **322.99**, p90 **26798.58**, p95 **37490.87**, p99 **53861.15**, max **108230.03**  
- D²TL: mean **744.70**, median **209.29**, p90 **2398.21**, p95 **3409.88**, p99 **4478.19**, max **6594.06**  
- tail_improvement: p95_vs_mlp=**0%**, p99_vs_mlp=**0%**, max_vs_mlp=**0%** (D²TL = MLP on tail in this run because Selector chose MLP for those samples).  

So in **this run**, D²TL tail equals MLP tail (no improvement). To align the paper:

- Either report these numbers and state that “in this run, tail percentiles for D²TL matched MLP (Selector chose MLP on worst-case samples); improvement is scenario-dependent,”  
- Or use a run/checkpoint where D²TL actually improves tail and then report that run’s numbers and improvement percentages.

**Suggested Table V (from actual run, physical-space MSE):**

| Model  | Mean   | Median | P90     | P95     | P99     | Max      |
|--------|--------|--------|---------|---------|---------|----------|
| MLP    | 764.33 | 202.05 | 2401.83 | 3409.88 | 4478.19 | 6594.06  |
| Mamba  | 6301.79| 322.99 | 26798.58| 37490.87| 53861.15| 108230.03|
| D²TL   | 744.70 | 209.29 | 2398.21 | 3409.88 | 4478.19 | 6594.06  |

Tail improvement (this run): 0% (D²TL tail = MLP tail).

---

## 11. Abstract and conclusion numbers

- **“6.8% MSE improvement”:** With code’s physical-space MSE, use **“up to 10.87% (extreme distance) and 8.91% (extreme compound)”** or keep 6.8% only if you adopt a different (e.g. scaled) MSE and a run that gives 6.8%.  
- **“within 2.1 dB/decade of Friis”:** Not supported by current run (errors −33.27 and −135.15). Either remove, or replace with “slope errors from current run: Rural −33.27 dB/decade, Urban −135.15 dB/decade” and discuss.  
- **“6.97–9.73 dB” rain:** Keep only if you add a sentence/script reference showing how this range is computed from the same data/code.  
- **“5.7× speedup”:** This run gives **5.89×**; “5.7×” is fine if from another setup; otherwise use 5.89×.  
- **“2.85 ms” early-exit:** This run gives **10.32 ms**; keep 2.85 ms only if you state it is from a different (e.g. lighter) configuration.

---

## 12. Summary of required edits for code/simulation consistency

1. **Equation (21):** Change to W_in ∈ ℝ¹⁰²⁴×²⁵⁶ (output 1024, split to 512 each).  
2. **Table I:** Replace with actual slope and slope_error_dB from your run (or re-run and use new results); or remove “within 2.1 dB/decade” and report the current errors.  
3. **Table II:** Either (a) switch to physical-space MSE and use the suggested Table II and improvement percentages above, or (b) define “MSE (scaled)” and add a script that outputs scaled MSE and use those values.  
4. **Table III:** Either use the suggested Table III from this run and add a hardware note, or keep current numbers and state they are from a different platform.  
5. **Table IV:** Use physical-space MSE values and ranks above, or scaled MSE from a new script; ensure “D²TL Selector best” and narrative match.  
6. **Table V:** Use the suggested Table V; adjust tail-improvement narrative to “0% in this run” or use a run where D²TL improves tail.  
7. **Abstract/Conclusion:** Align “6.8%”, “2.1 dB/decade”, “6.97–9.73 dB”, “5.7×”, “2.85 ms” with the chosen tables and run (or add a single “reference run” footnote).

After these edits, the paper will be **fully consistent with the provided code and the simulation/output of `run_all_experiments_via_api.py` and `all_experiment_results_via_api.json`.**
