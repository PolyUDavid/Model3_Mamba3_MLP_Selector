# GITHUB/code — 模型、API、仿真、Dashboard、实验脚本

本目录包含 D²TL 的**完整可运行代码**，便于复现与提交 Git。

## 目录说明

| 目录/文件 | 内容 |
|-----------|------|
| **mlp_service/** | 主路径模型与 API：`model.py`（CoverageMLP）、`api.py`（FastAPI 8001） |
| **mamba_service/** | 物理备份 API：`api.py`（FastAPI 8002），加载 `../models/mamba3_coverage.py` |
| **selector_brain/** | 编排层：`selector.py`（FastAPI 8000），同时调用 MLP 与 Mamba |
| **models/** | Mamba-3 骨干：`mamba3_coverage.py`（SelectiveSSM + MambaBlock ×8） |
| **data_generator/** | 训练数据生成：`generate_coverage_data_v2.py`、`generate_coverage_data.py` |
| **training/** | 训练脚本：`train_mlp.py`（MLP）、`train_coverage.py`（Mamba-3） |
| **simulation/** | Pygame 仿真：`pygame_simulation.py`（18 RSU、本地/API 双模式），见 simulation/README.md |
| **dashboard/** | Streamlit 指挥中心：`app.py`、`.streamlit/config.toml`，见 dashboard/README.md |
| **experiments/** | 七组实验：`run_all_experiments.py`、`run_all_experiments_via_api.py`、`validate_experiment_data.py`、EXPERIMENT_REPORT.md |
| **start_all_services.sh** | 一键启动 MLP(8001)、Mamba(8002)、Selector(8000)、Dashboard(8501) |
| **generate_*_backbone*.py** | 架构图脚本：MLP / Mamba3 / D2TL 全架构 PNG，输出可放 GITHUB/data/figures/backbones/ |

## 路径与运行说明

- **在仓库根目录运行**（推荐）：服务与训练脚本的 `BASE_DIR`/`PARENT_DIR` 指向 `Model_3_Coverage_Mamba3`，请使用根目录下的 `d2tl/`、`training/`、`models/` 运行。
- **仅用本 GITHUB 包运行**：从本 repo 根目录执行时，可将 `GITHUB/code` 视为 `d2tl`+`models`+`training` 的并集；MLP 服务需在 `code/mlp_service` 下运行（含 `model.py`），Mamba 服务需能访问 `code/models/mamba3_coverage.py`（当前 api 中 `PARENT_DIR = parent.parent`，即 repo 根上一级为 `Model_3_Coverage_Mamba3` 时正确；若只拷贝 GITHUB 文件夹，则需把 `PARENT_DIR` 改为 `Path(__file__).resolve().parent.parent` 并保证其下存在 `models/` 与 `training/`）。

## 快速运行（在仓库根目录）

```bash
# 一键启动 API + Dashboard
bash d2tl/start_all_services.sh

# Pygame 仿真（接 API 时先启动上面）
python3 d2tl/simulation/pygame_simulation.py --api

# 训练 MLP
python3 d2tl/training/train_mlp.py

# 训练 Mamba-3（需先有 coverage 数据）
python3 training/train_coverage.py

# 七组实验（本地 或 Via API）
python3 d2tl/experiments/run_all_experiments.py
python3 d2tl/experiments/run_all_experiments_via_api.py

# 生成数据（V2）
python3 data_generator/generate_coverage_data_v2.py
```

本目录与 `d2tl/`、`models/`、`training/`、`data_generator/` 对应，为 **GITHUB 打包用** 的完整代码副本（含 Pygame、Dashboard、实验脚本与架构图脚本）。
