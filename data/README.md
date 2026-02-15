# GITHUB/data — 实验、训练与图表数据包

本目录包含**所有实验与训练的 JSON** 以及**论文/复现用图表**，便于复现与提交 Git。

## experiment/ — 所有实验 JSON

| 文件 | 说明 |
|------|------|
| **all_experiment_results.json** | D²TL 七组实验（本地运行，Scaled MSE） |
| **all_experiment_results_via_api.json** | D²TL 七组实验（Via 真实 API，物理量 MSE/延迟） |
| **temporal_results.json** | 时序/序列实验 |
| **temporal_experiment_results.json** | 时序实验补充 |
| **first_principles_validation.json** | 第一性原理验证 |
| **MANIFEST.txt** | 各文件说明及原始路径 |

## training/ — 所有训练 JSON

| 文件 | 说明 |
|------|------|
| **mlp_training_history.json** | D²TL MLP 训练曲线（epochs, train_loss, val_loss, val_mae） |
| **mlp_service_training_history.json** | MLP 服务对应训练历史 |
| **mamba_training_history.json** | Mamba-3 训练曲线（Complete Package） |
| **training_history_v2_final.json** | Mamba V2 最终训练历史 |
| **training_history_training_dir.json** | training/ 目录下 Mamba 训练历史 |
| **MANIFEST.txt** | 各文件说明及完整 train/val/test 大文件路径（在仓库其他位置） |

完整 train/val/test 大体积 JSON 路径见 **training/MANIFEST.txt**。

## figures/ — 论文与复现用图表

| 子目录 | 内容 |
|--------|------|
| **figures/backbones/** | 架构图：MLP_backbone_architecture.png、Mamba3_backbone_architecture.png、D2TL_full_architecture.png（Selector+MLP+Mamba3） |
| **figures/visualizations/** | 训练与实验图：01_training_loss_and_val_mae.png、02_exp1_distribution.png … 09_temporal_results.png（与 paper_package/07_visualizations/plots/ 一致） |

图表可由 `paper_package/07_visualizations/plot_all_training_and_experiments.py` 与 `paper_package/08_backbones/generate_*.py` 重新生成；本目录为打包副本。
