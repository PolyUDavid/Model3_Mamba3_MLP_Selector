# 文档与代码校对报告 (Verification Report)

**校对日期:** 2026-02  
**校对范围:** ALGORITHMS_AND_EXPERIMENTS.md、训练/实验数据定义、MLP/Mamba-3 backbone 与代码一致性

---

## 1. 数据生成 (Physics & Data Generation) — 一致 ✅

| 项目 | 文档 (F1–F59) | 代码 `generate_coverage_data_v2.py` | 结论 |
|------|----------------|-------------------------------------|------|
| 常数 f_c, P_tx, N_0, B, n, d0, σ_sh, G_max, θ_bw, P_min, γ_min | 与 2.1 节一致 | `CARRIER_FREQ=5.9`, `TX_POWER=33`, `NOISE_FLOOR=-95`, `BANDWIDTH=10`, `PATH_LOSS_EXPONENT=3.5`, `REFERENCE_DISTANCE=1.0`, `SHADOWING_STD=8`, `ANTENNA_GAIN_MAX=10`, `ANTENNA_BEAMWIDTH=120`, `MIN_RX_POWER=-90`, `MIN_SINR=-5` | 一致 |
| η_obs(k), A_weather(w) | F4, F6 | `obstacle_factor` {0:1,1:1.2,2:1.5,3:2}, `weather_attenuation` {0:0,1:2,2:5,3:8} | 一致 |
| 路径损耗 L0 + 10·n_eff·log10(d/d0) + shadowing | F7–F9 | `_calculate_path_loss`: L0=20*log10(4π*d0/λ), path_loss += 10*exponent*log10(d/d0), path_loss += normal(0,σ), max(40) | 一致 |
| 天线增益 (角度差、cos² 内 beam、0.01 外 beam) | F10–F12 | `_calculate_antenna_gain`: angle_diff 公式一致，cos² 内、0.01 外 | 一致 |
| 接收功率 P_tx+G−PL−A_weather | F13 | `received_power_dbm = tx_power_dbm + antenna_gain_db - path_loss_db - weather_attenuation` | 一致 |
| 干扰源数 floor(ρ/50)−1 | F14 | `num_interferers = max(0, int(vehicle_density/50)-1)` | 一致 |
| 干扰功率 (α_i, path_loss_diff, 10^((P_rx−ΔPL)/10)) | F15–F18 | `interferer_distance_factor` U(1.5,3), `path_loss_diff = 10*n*log10(α)`, `10^((signal - path_loss_diff - u)/10)` | 一致 |
| SINR 线性与 dB | F20–F23 | `_calculate_sinr`: signal_linear, interference_linear, noise_linear, 10*log10(signal/(I+N)) | 一致 |
| 覆盖半径 (PL_max, L0, log_distance, shadowing margin, clip 200–1000) | F24–F32 | `effective_min_rx=-85+U(-5,5)`, `interference_margin=1.5*N_int`, `10^((max_path_loss-L0)/(10*n_eff))`, 0.5*σ_sh, clip(200,1000) | 一致 |
| SINR/环境因子、最终半径 clip(150,1000) | F33–F36 | `sinr_factor`, `environment_factor`, `coverage_radius_m *= ...`, `clip(150, 1000)` | 一致 |
| 面积、覆盖概率、QoS 四分量、Throughput | F37–F48 | `coverage_area_km2`, `_sinr_to_coverage_prob`, `_calculate_qos`, `_estimate_throughput` 公式一致 | 一致 |
| 采样范围 (RSU 0–2000, TX 30–36, tilt 0–15, azimuth 6 值, density 0–3, weather p, vehicle lognormal, d 10–1000, angle 0–360, h_rx=1.5, f=5.9) | F49–F59 | `_generate_single_sample` 中对应采样一致 | 一致 |

---

## 2. MLP Backbone — 一致 ✅（代码注释 1 处修正）

| 项目 | 文档 (F60–F80) | 代码 `d2tl/mlp_service/model.py` | 结论 |
|------|----------------|-----------------------------------|------|
| 输入 13，隐藏 256，输出 5 | F60–F73 | `input_dim=13`, `hidden=256`, 5 个 head 各 Linear(256,1) | 一致 |
| 第 0 层 Linear(13,256)+LayerNorm+GELU，无 Dropout | F60–F64 | `layers = [Linear(13,256), LayerNorm, GELU]` | 一致 |
| 第 1–7 层 Linear(256,256)+LayerNorm+GELU+Dropout(0.1) | F65–F66 | `for _ in range(n_layers-1):` 加 Linear, LN, GELU, Dropout | 一致 |
| 5 个 head，Xavier gain=0.5，bias=0 | F74–F76 | `_init_weights`: xavier_uniform gain=0.5, zeros bias | 一致 |
| 参数量 | F77–F80 合计 ≈469K | `get_num_params()` 实际约 469K | 一致 |
| 代码文件内注释 | — | 原写 “Params ≈ 1.3M” | **已改为 ≈469K**（见下方修正） |

---

## 3. Mamba-3 Backbone — 2 处维度修正 ✅

| 项目 | 文档 | 代码 `models/mamba3_coverage.py` | 结论 |
|------|------|----------------------------------|------|
| 输入投影 Linear(13,256)+LN+GELU+Dropout | F81–F82 | `input_proj = Sequential(Linear(13,256), LN, GELU, Dropout)` | 一致 |
| MambaBlock in_proj | 原 F85: “W ∈ R^{512×256}”；原 F86: “x_ssm, g ∈ R^256” | `in_proj = Linear(d_model, d_inner*2)` = Linear(256, **1024**)；`chunk(..., 2)` → 各 **512** | **文档已修正**：in_proj 输出 1024，split 后 x_ssm、g 各 512 |
| SelectiveSSM 输入维度 | — | `SelectiveSSM(self.d_inner, ...)` = 512 | 与修正后 F86 一致 |
| MambaBlock out_proj | 原 F89: “W_out ∈ R^{256×256}” | `out_proj = Linear(d_inner, d_model)` = **512→256** | **文档已修正**：W_out ∈ R^{256×512} |
| A_log, D, x_proj, dt_proj, conv, out_proj (SSM 内) | F91–F102, Appendix E 13.3 | d_inner=512：Conv1d 512, x_proj 512→1536, dt_proj 512→512, A_log 512×16, D 512 | 一致 |
| 8 层 MambaBlock，最后 LN，pool，5 heads | F103–F107 | `n_layers=8`, `norm_f`, mean/squeeze, 5×Linear(256,1) | 一致 |
| d_model=256, n_layers=8, d_state=16, d_conv=4, expand=2, d_inner=512 | F108 | 构造函数默认值一致 | 一致 |

---

## 4. Selector Brain — 一致 ✅

| 项目 | 文档 (F110–F131) | 代码 `d2tl/selector_brain/selector.py` | 结论 |
|------|-------------------|----------------------------------------|------|
| 常数 d_th=500, d_th2=700, Δ_div=5 | F110 | `DISTANCE_THRESHOLD=500`, 700 在分支中, `DIVERGENCE_THRESHOLD_DB=5.0` | 一致 |
| 触发累加 (d>700 +0.35; d>500 +0.20; d>500&w≥2 +0.20; w≥3 +0.10; d>500&k≥2 +0.15; nint≥3&d>400 +0.10; n_risk≥3 +0.15; min(s,1)) | F111–F120 | `analyze()` 中对应 if/elif 与累加一致 | 一致 |
| use_mamba = (s≥0.3) | F121 | `'use_mamba': score >= 0.3` | 一致 |
| 发散 ΔP, Δγ, ΔR；divergent = (ΔP>5 or Δγ>5) | F122–F125 | `check_divergence`: power_diff, sinr_diff, radius_diff; `divergent = power_diff > 5 or sinr_diff > 5` | 一致 |
| use_mamba_final = use_mamba or (divergent and s≥0.15) | F126–F127 | `use_mamba = analysis['use_mamba'] or (divergence['divergent'] and analysis['trigger_score'] >= 0.15)` | 一致 |
| risk_level EXTREME/ELEVATED/NORMAL | F128 | `'EXTREME' if score >= 0.5 else 'ELEVATED' if score >= 0.3 else 'NORMAL'` | 一致 |

---

## 5. 训练 (归一化、目标缩放、损失、划分) — 一致 ✅

| 项目 | 文档 (F132–F161) | 代码 train_mlp.py / train_coverage.py | 结论 |
|------|-------------------|----------------------------------------|------|
| 特征归一化 μ, σ, (x−μ)/σ | F132–F134 | `feature_mean`, `feature_std`, `(features - mean) / std` | 一致 |
| 目标缩放 Power (+260)/230, SINR (+170)/230, Radius (−150)/90, Area (−0.07)/0.12, QoS/100, clip [0,1] | F135–F140 | 两脚本中相同公式与 clip | 一致 |
| 损失权重 0.15+0.15+0.30+0.30+0.10 | F141–F142 | `loss_power*0.15 + loss_sinr*0.15 + loss_radius*0.30 + loss_area*0.30 + loss_qos*0.10` | 一致 |
| 惩罚项 radius/area≥0, QoS∈[0,1] | F143–F147 | ReLU(−pred)*0.05 等，与文档一致 | 一致 |
| MLP: AdamW lr=1e-3, weight_decay=0.01; Cosine T_max=150, eta_min=1e-5 | F148–F149 | `AdamW(..., lr=1e-3, weight_decay=0.01)`, `CosineAnnealingLR(..., T_max=150, eta_min=1e-5)` | 一致 |
| Mamba: warmup 0–9, stable 10–99, decay 100–149; base 1e-4, min 1e-6 | F150–F152 | `get_lr_schedule`: warmup_epochs=10, stable_epochs=90, base_lr=1e-4, min_lr=1e-6 | 一致 |
| 梯度裁剪 max_norm=1 | F153 | `clip_grad_norm_(model.parameters(), 1.0)` | 一致 |
| R² / MAE 定义与 overall | F154–F159 | `compute_metrics` 中 SS_res, SS_tot, R2, MAE 一致 | 一致 |
| 划分 Train 21000, Val 4500, Test 4500（或 rest）；seed 42 | F160–F161 | `train_size=21000`, `val_size=4500`, `test_size=4500` 或 rest；`manual_seed(42)` | 一致 |

---

## 6. 实验设计 (Exp1–Exp7, Batch) — 一致 ✅

| 项目 | 文档 (F162–F212) | 代码 run_all_experiments_via_api.py / run_all_experiments.py | 结论 |
|------|-------------------|--------------------------------------------------------------|------|
| Exp1 极端定义、计数、weather/density/distance/type | F162–F170 | 与 exp1_distribution 逻辑一致 | 一致 |
| Exp2 距离 60 点、理论斜率 −10·n_eff、线性拟合、slope_error | F171–F176 | `np.linspace(50,1000,60)`, theory_slope, polyfit log_d, slope_error_dB | 一致 |
| Exp3 d=300, 两密度 Rural/Suburban, 4 天气, physics/mlp/mamba/dual/trigger | F177–F180 | test_dist=300, (0,1), w in range(4), physics_power 公式 | 一致 |
| Exp4 测试集 4500、分类、MSE 与 mlp_improvement | F181–F186 | test_data = data[-4500:], categories, _mse_one, mlp_improvement | 一致 |
| Exp5 50 次计时、params、parallel、early-exit r=0.15、speedup | F187–F192 | 50× POST, health params, dual_effective = 0.15*max + 0.85*mlp_t | 一致 |
| Exp6 六策略 MSE（MLP Only, Mamba Only, Random 50/50, Any-Extreme, D²TL, Soft Blend） | F193–F199 | variants 与 _mse_one 一致 | 一致 |
| Exp7 误差分布 mean/median/p90/p95/p99/max、tail_improvement | F200–F205 | errs_mlp/mamba/dual, percentile, tail_improvement 公式 | 一致 |
| Batch N_test=4500, B_batch=80, K=57 | F206–F208 | `BATCH_SIZE=80`, `test_data = data[-4500:]`, 57 batches | 一致 |
| 实验用物理常数 | F209–F212 | CARRIER_FREQ_GHZ, TX_POWER_DBM, PATH_LOSS_EXPONENT, DENSITY_OBSTACLE, WEATHER_ATTEN, ANTENNA_GAIN | 一致 |

---

## 7. 其他文档 (DATA / EXPERIMENTS / TRAINING / ARCHITECTURE / API) — 说明

- **DATA.md / TRAINING.md:** 数据路径以 `paper_package/01_training_data/MANIFEST.txt` 及 `Model_3_Complete_Package/training_data_package/` 为准；与 `training_data/coverage_training_data_v2.json` 等实际加载路径通过 MANIFEST 对应，**定义一致**。
- **EXPERIMENTS.md:** Exp1–Exp7 描述与 ALGORITHMS_AND_EXPERIMENTS.md 及脚本一致。其中 **Exp5 的延迟数值**（如 “~2.85 ms”“~16.15 ms”）可能来自不同运行环境；以实际运行 `run_all_experiments_via_api.py` 得到的 `all_experiment_results_via_api.json` 为准（例如 MLP ~1.4 ms、Mamba ~60.8 ms、early-exit ~10.3 ms）。
- **ARCHITECTURE.md / API_AND_SERVICES.md:** 与 backbone、Selector、端口、端点描述一致。

---

## 8. 已做修正汇总

1. **ALGORITHMS_AND_EXPERIMENTS.md**  
   - **F85:** 改为 in_proj 输出维度 `(d_inner×2)×d_model`，即 **1024×256**（或写为 Linear(256→1024)）。  
   - **F86:** 改为 split 后 **x_ssm, g ∈ R^{d_inner} = R^{512}**（不是 R^256）。  
   - **F89:** 改为 **W_out ∈ R^{256×512}**（out_proj: d_inner→d_model，即 512→256）。  
   - **Appendix E 13.2:** MambaBlock in_proj 写为 256→**1024**（d_inner×2），split 后各 512。

2. **d2tl/mlp_service/model.py**  
   - 文件头注释中 “Params ≈ 1.3M” 改为 **“Params ≈ 469K”**，与 F77–F80 及实际 `get_num_params()` 一致。

---

## 9. 结论

- **数据公式、训练数据与实验数据定义、两个 backbone 的数学描述**与代码实现**一致**。  
- 上述**维度与注释**已按本报告修正，文档与代码已对齐。  
- 建议：之后若修改数据生成、模型结构或实验脚本，请同步更新 `ALGORITHMS_AND_EXPERIMENTS.md` 与本 `VERIFICATION_REPORT.md`。
