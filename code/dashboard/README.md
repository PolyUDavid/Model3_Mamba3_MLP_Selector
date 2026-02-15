# D²TL Command Center — Streamlit Dashboard

本目录为 **D²TL 指挥中心** Streamlit 应用的代码副本。在仓库根目录下运行：

```bash
# 先启动三个 API（Selector 8000、MLP 8001、Mamba 8002）
bash d2tl/start_all_services.sh

# 或单独启动 Dashboard（需 API 已运行）
streamlit run d2tl/dashboard/app.py --server.port 8501
```

**功能**：Mission Control、覆盖地图、实时预测、场景 Playbook、告警、事件日志、RSU 列表、系统健康、实验/训练/物理报告、设置等。

**配置**：`dashboard/.streamlit/config.toml` 为主题与服务器选项。

**说明**：本 GITHUB 包内文件与 `d2tl/dashboard/` 一致；实际运行请使用仓库根目录下的 `d2tl/dashboard/app.py`。
