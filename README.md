# Raha + Splink 原库诊断工程 V2

本工程用于继续分析已实际成功运行的 Raha 1.26 / Splink 5.0 原版库，不是其性能的宣传指标。

## V1 已知事实（来自用户上传 GitHub Actions 原始 JSON）
- Raha：220x5，42 个真实错误单元格；只启用 PVD、RVD，模拟人工标签预算为 20 行；TP=33，FP=0，FN=9，召回率 78.57%。
- Splink：400 条企业记录，80 对真实重复；0.5 阈值下 TP=55，FP=71，FN=25。Splink 日志明确显示名称字段部分 `m` 参数未训练。
- 对 Splink 原工作流所使用 `region+sector OR bank_suffix` 阻断规则，离线复核：79800 个可能配对，筛到 762 个候选对，其中 70 个真重复进入候选、10 个在阻断阶段丢失；还有 15 个真重复在评分/阈值阶段未被接受。

## 操作
1. 将这个目录的所有内容复制到你的 GitHub 仓库根目录（包括 `.github` 隐藏目录）。
2. 打开 GitHub Actions，运行 **Original Raha and Splink diagnostic verification V2**。
3. 下载两个 Artifacts：`raha-diagnostic-v2`、`splink-diagnostic-v2`。
4. 如失败，下载上传的日志即可定位；不要把绿色工作流当成算法达到生产效果。

## 本地只审计样本，无需安装 Raha 和 Splink
```bash
python scripts/audit_fixture.py
```

## V2 新增检查
- Raha：相同原版 API，分别运行 10、20、40 行模拟人工标注，按字段记录命中、漏报、误报；**只运行已经验证的 PVD/RVD 两类**，暂未测试其他算法族。
- Splink：保留 V1 已运行成功的设置、模型训练调用，在 0.5 阈值复现结果；额外报告候选生成召回与 0.01、0.1、0.3、0.5、0.7、0.9 的阈值表现。
- 注意：不同运行可能因抽样与训练随机性出现波动；对最终正式选型仍需多批次、真实人工真值数据与规模化测试。

## 安全
仅含虚构测试数据。不要把客户真实数据上传到公共 GitHub Actions。生产环境不能将真实客户数据、含敏感内容的错误日志直接上传到第三方 CI。
