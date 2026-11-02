# 数据问题发现算法测试 V3（2026-11-02）

## 真实执行情况

- 已在 Python 3.13.5 本地运行 `benchmark_v3.py`、`stress_blocking.py`、`verify_v3.py`。
- 采用了已安装的 NumPy / pandas / scikit-learn / RapidFuzz，**没有调用 Raha 或 Splink 原版库**。
- 试图通过 pip 下载 `splink==5.0.0`、`raha`，但包索引 DNS 解析失败：`Temporary failure in name resolution`。
- `optional_original_libs/` 中保存上轮编写但未执行的原库实验脚本。它们是**未验收草稿**，不能视为官方库运行成功。
- 所有数据纯属合成，真实答案位于专用 JSON 中，未传给匹配模型。

## 运行命令

```bash
python -m pip install -r requirements.txt
python benchmark_v3.py
python stress_blocking.py
python verify_v3.py
```

## 原库安装与后续验证（联网环境）

```bash
python -m pip install 'splink==5.0.0' pandas
python -m pip install raha
```

建议在单独虚拟环境中检查版本和依赖。`raha` 的旧依赖可能不兼容 Python 3.13。**原库验证脚本必须先根据真实安装版本修正接口并运行，不应直接作为生产代码。**

## 主要数据

见 `results/v3_report.json`、`results/blocking_summary.csv`、`results/cold_start_summary.csv` 和 `results/stress_blocking.json`。

1. 400条实体记录、80组真重复：原样数据的完整阻断方案召回率100%，但这是合成数据；人为隐藏12组重复记录的所有可用匹配信号后，降至85%。
2. 100条无关记录共享名称前缀后，候选对从1201增至5896，召回率也从100%降至95%。
3. 30次不同实体划分中，原阻断平均召回86.42%；附加名称前缀在这份合成数据上达到100%，**不是外部数据集上的保证**。30次划分共享同一合成数据，彼此并非独立数据集。
4. 1,500条订单中，101条是已注入的可观察脏记录，另有15条缺少参考来源的事实错误。20次随机启动中，14次首次8条标注没有正例。
5. 无先验指定正例，每次总共32条人工标签的情况下：随机标注平均召回34.00%，不确定性标注平均召回35.67%；二者均有3/20次无法训练二分类器。
6. 排除某类错误参与训练时，在测试集该类4条错误上的召回为0/4（样本数过少，不能泛化）。

## 本次暴露的架构要求

- 候选生成、匹配评分、人工复核必须拆开；评测 `candidate_generation_recall`。
- 冷启动应支持无正例时的流程，并独立报告哪些问题类型完全未被识别。
- 不能用合成数据集内100%结果宣布商业可用；必须做跨来源、跨批次、人工真值独立验证。
- 不能让不可靠的匹配模型直接合并记录或自动修复。
- 原库验证仍未完成，需要在可下载依赖的环境继续进行。
