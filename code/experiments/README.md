# D²TL Experiments — 七组实验与 Via-API 复现

本目录包含实验脚本与说明的副本。

| 文件 | 说明 |
|------|------|
| **run_all_experiments.py** | 本地运行全部 7 组实验（不依赖 API），输出 `results/all_experiment_results.json` 及 plots |
| **run_all_experiments_via_api.py** | 通过真实 Selector/MLP/Mamba API 运行 7 组实验，输出 `results/all_experiment_results_via_api.json` |
| **validate_experiment_data.py** | 校验 `all_experiment_results_via_api.json` 与模型逻辑一致性 |
| **EXPERIMENT_REPORT.md** | 实验设计、指标与结果说明 |

**运行方式（从仓库根目录）**：

```bash
# 方式一：本地实验（无需启动 API）
python3 d2tl/experiments/run_all_experiments.py

# 方式二：Via API（需先启动 8000/8001/8002）
python3 d2tl/experiments/run_all_experiments_via_api.py
```

**结果位置**：  
- 实验 JSON：`d2tl/experiments/results/` 与 **GITHUB/data/experiment/**（含 `all_experiment_results.json`、`all_experiment_results_via_api.json`）  
- 图表：`paper_package/07_visualizations/plots/` 与 **GITHUB/data/figures/visualizations/**
