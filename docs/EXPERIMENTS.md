# Experiments — What I Ran and What I Found (First-Person)

I designed and ran seven main experiments to validate the D²TL system and both backbones. Here I summarize what each experiment does and where the results live.

---

## Experiment 1: Scenario Distribution

I analyzed the composition of my evaluation set (30K samples) by weather, density, distance, and extreme types. I use this to interpret where “normal” vs “extreme” regimes sit.

- **Result**: 80.6% extreme (long distance, dense urban, heavy weather, or high interference); 19.4% normal.
- **Data**: `paper_package/02_experiment_data/all_experiment_results.json` or **GITHUB/data/experiment/** (all_experiment_results.json, all_experiment_results_via_api.json) → `exp1`.
- **Plot**: `paper_package/07_visualizations/plots/02_exp1_distribution.png`.

---

## Experiment 2: Distance–Power Decay (Friis Consistency)

I checked whether MLP, Mamba, and D²TL follow the theoretical path-loss slope (Rural n=3.5, Urban n=5.25). I report slope errors in dB/decade.

- **Result**: D²TL slope error is documented (Rural/Urban vs theory); Mamba is used in the dual system to improve long-range behavior.
- **Data**: `all_experiment_results.json` → `exp2` (e.g. Rural_theory_slope, Rural_mlp_slope, Rural_mamba_slope, Rural_dual_slope, and Urban counterparts).
- **Plot**: `07_visualizations/plots/03_exp2_slopes.png`.

---

## Experiment 3: Rainstorm Coverage Collapse

I compared physics (theory), MLP, Mamba, and D²TL under Clear / Light / Moderate / Heavy Rain across Rural, Suburban, Urban. I focus on whether rain attenuation is learned in a physically plausible range.

- **Result**: Mamba learns rain attenuation in the ~6.97–9.73 dB range (theory 8 dB); D²TL uses Mamba when trigger fires (e.g. heavy rain).
- **Data**: `all_experiment_results.json` → `exp3` (nested by density and weather).
- **Plot**: `07_visualizations/plots/04_exp3_rain_coverage.png`.

---

## Experiment 4: Stratified Performance (Normal vs Extreme)

I split evaluation into normal, extreme_weather, extreme_distance, extreme_density, and extreme_compound, and reported MSE for MLP, Mamba, and D²TL.

- **Result**: D²TL achieves the best MSE in every stratum. Largest gains: extreme_distance (+6.2%) and extreme_compound (+6.8%).
- **Data**: `all_experiment_results.json` → `exp4`.
- **Plot**: `07_visualizations/plots/05_exp4_stratified_mse.png`.
- **Report**: `d2tl/experiments/EXPERIMENT_REPORT.md`.

---

## Experiment 5: Budget / Cost (Latency and Params)

I measured inference latency and parameter count for MLP-only, Mamba-only, D²TL parallel, and D²TL early-exit.

- **Result**: D²TL early-exit ~2.85 ms vs Mamba-only ~16.15 ms (≈5.7× speedup); MLP ~0.5 ms, 469K params; Mamba ~13.7M params.
- **Data**: `all_experiment_results.json` → `exp5`.
- **Plot**: `07_visualizations/plots/06_exp5_latency_params.png`.

---

## Experiment 6: Ablation (Strategy Comparison)

I compared MSE across: MLP only, Mamba only, random 50/50, any-extreme trigger, D²TL Selector, and soft blend. This shows that my Selector policy is better than naive strategies.

- **Result**: D²TL Selector matches or beats MLP-only and clearly outperforms Mamba-only and the other baselines.
- **Data**: `all_experiment_results.json` → `exp6`.
- **Plot**: `07_visualizations/plots/07_exp6_ablation.png`.

---

## Experiment 7: Tail-Risk Analysis

I computed mean, median, p90, p95, p99, and max error for MLP, Mamba, and D²TL to check that D²TL does not worsen tail behavior.

- **Result**: D²TL matches MLP’s tail (p95, p99, max) while adding physics backup when needed.
- **Data**: `all_experiment_results.json` → `exp7`.
- **Plot**: `07_visualizations/plots/08_exp7_tail_risk.png`.

---

## How I Run the Experiments

I run all seven experiments with one script (no need to start the APIs):

```bash
python d2tl/experiments/run_all_experiments.py
```

It loads MLP and Mamba from checkpoints, runs the evaluation logic, writes `d2tl/experiments/results/all_experiment_results.json`, and can generate plots in `d2tl/experiments/results/plots/`. I also regenerate the paper-package plots (including Exp1–Exp7) with:

```bash
python paper_package/07_visualizations/plot_all_training_and_experiments.py
```

---

## Other Data I Provide

- **Temporal / sequence experiment**: `paper_package/02_experiment_data/temporal_results.json`, `temporal_experiment_results.json`. Plot: `07_visualizations/plots/09_temporal_results.png`.
- **First-principles validation**: `paper_package/02_experiment_data/first_principles_validation.json`.

Together, these experiments cover scenario mix, physics consistency, rain behavior, stratified performance, cost, ablation, and tail risk—giving a full picture of how my dual-path design and both backbones behave.
