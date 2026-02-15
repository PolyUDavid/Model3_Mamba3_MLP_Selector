# Pygame Simulation — Live Dual-Path Visualization

本目录为 **D²TL Pygame 仿真** 的代码副本。在仓库根目录下运行（需先安装 `pygame`）：

```bash
# 从仓库根目录 Model_3_Coverage_Mamba3 执行
pip install pygame

# 本地模式（加载本地 MLP + Mamba 权重）
python3 d2tl/simulation/pygame_simulation.py

# 接真实 API（需先启动 Selector 8000、MLP 8001、Mamba 8002）
python3 d2tl/simulation/pygame_simulation.py --api
# 或
D2TL_USE_API=1 python3 d2tl/simulation/pygame_simulation.py
```

**功能**：车辆沿路径行驶，地图上 18 个 RSU；根据距离/天气/密度实时调用 Selector，显示 MLP（蓝）或 Mamba（红）决策、Trigger、Power、QoS 等。

**按键**：空格 暂停 | R 重置 | 1–4 强制天气 | 0 自动天气 | ESC 退出

**说明**：本 GITHUB 包内文件与 `d2tl/simulation/pygame_simulation.py` 一致，便于打包与引用；实际运行请使用仓库根目录下的 `d2tl/simulation/`。
