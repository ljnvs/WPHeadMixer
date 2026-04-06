# WPHeadMixer：代码、数据与证据发布仓库

[English Version](README.md)

本仓库是 WPHeadMixer 项目的公开发布版本：

**WPHeadMixer: A Wavelet-Patch Forecasting Framework with Modular Branch Heads for Long-Term Time Series Forecasting**

该仓库面向公开发布与复现支持，保留了以下内容：

- 查看公开代码实现
- 直接使用随仓库提供的六个 benchmark 数据集运行保留的复现实验脚本
- 将论文中的主要结论对应到仓库中的证据摘要文件与复现脚本

本仓库是一个**面向发布的精简快照**，不是完整的私有研究工作区镜像。

## 仓库结构

- [code/wpheadmixer/](code/wpheadmixer/)：公开代码与发布版复现实验脚本
- [code/wpheadmixer/data/](code/wpheadmixer/data/)：随仓库提供的六个 benchmark 数据集
- [evidence/](evidence/)：支撑论文主要结论的摘要级证据文件
- [paper_evidence_map.md](paper_evidence_map.md)：论文主要结论/表格与仓库证据的对应关系
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)：最小发布检查清单

## 环境配置

建议使用 **Python 3.10**。

在仓库根目录执行：

```bash
pip install -r code/wpheadmixer/requirements.txt
```

补充说明：

- `thop` 不是必需依赖，只有在你需要计算 FLOPs 时才需要额外安装。

主实验入口：

- [code/wpheadmixer/run_LTF.py](code/wpheadmixer/run_LTF.py)

## 数据集

本仓库已经随代码提供这六个 benchmark 数据集：

- `ETTh1`
- `ETTh2`
- `ETTm1`
- `ETTm2`
- `Weather`
- `Exchange`

数据文件位于 [code/wpheadmixer/data/](code/wpheadmixer/data/)。

各数据文件来源口径如下：

- `ETTh1`、`ETTh2`、`ETTm1`、`ETTm2`：ETT（Electricity Transformer Temperature）benchmark
- `weather.csv`：常用长序列预测 benchmark 中的 Weather 数据
- `exchange_rate.csv`：常用长序列预测 benchmark 中的 Exchange Rate 数据

更详细的数据目录和脚本说明见：

- [code/wpheadmixer/readme.md](code/wpheadmixer/readme.md)
- [code/wpheadmixer/data/README.md](code/wpheadmixer/data/README.md)

## 本发布包支持什么

本仓库主要用于支持：

- 检查公开代码实现
- 直接运行保留的论文复现实验脚本
- 重新生成保留的摘要级证据文件
- 核对论文中的主要结论对应仓库中的哪些证据材料

本仓库**不包含**：

- 论文正文源文件
- 完整实验转储
- 全量日志集合
- 模型 checkpoint
- 大型中间产物
- 本地缓存文件
- 与本次发布无关的基线源码树

## 已保留证据

当前保留的摘要级证据文件包括：

- [evidence/main_results/linear_main_results_summary.md](evidence/main_results/linear_main_results_summary.md)
- [evidence/multi_seed/small_multi_seed_verification_summary.md](evidence/multi_seed/small_multi_seed_verification_summary.md)
- [evidence/ablation/five_seed_head_ablation_summary.md](evidence/ablation/five_seed_head_ablation_summary.md)
- [evidence/wavelet_sensitivity/wavelet_sensitivity_summary.md](evidence/wavelet_sensitivity/wavelet_sensitivity_summary.md)
- [evidence/interpretability/etth2_interpretability_summary.md](evidence/interpretability/etth2_interpretability_summary.md)
- [evidence/interpretability/ettm1_interpretability_summary.md](evidence/interpretability/ettm1_interpretability_summary.md)

如需快速查看“论文结论与仓库证据的对应关系”，请直接看：

- [paper_evidence_map.md](paper_evidence_map.md)

## 发布说明

- 本发布包由更大的本地研究工作区整理而来。
- 已保留的证据文件在公开前做了轻量清洗，去除了本地机器路径。
- 当前版本只保留理解方法、检查代码、使用六个核心 benchmark 数据集和核对证据所需的核心内容。

## 许可与第三方说明

- 项目代码保留许可文件：[code/wpheadmixer/LICENSE](code/wpheadmixer/LICENSE)
- vendored 的小波依赖代码位于 [code/wpheadmixer/pytorch_wavelets/](code/wpheadmixer/pytorch_wavelets/)，其目录中也保留了对应许可证文件。
