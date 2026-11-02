# 原版 Raha / Splink 实测方案（等待在线 CI 运行）

**状态：未运行原库。** 当前 ChatGPT 容器无法安装 Raha 和 Splink，本工程只验证了 YAML/代码语法，不能把此前的 scikit-learn/RapidFuzz 数字当作原库结果。

## GitHub Actions 操作
1. 创建独立的 GitHub 测试仓库（可以是私有仓库；这份包只有虚构测试数据）。
2. **解压后将文件内容放到仓库根目录**。注意 `.github` 是隐藏目录，别漏掉。
3. 提交、推送后在仓库的 **Actions → Original Raha and Splink library verification → Run workflow** 手动触发。
4. 两个 Job 分别安装真正的 Raha（Python 3.10）及 Splink 5（Python 3.11），调用 `optional_original_libs/` 下的脚本。
5. 查看 Job 的日志、结束状态，下载 artifacts 中的 `raha_library_result.json` / `splink5_library_result.json`。如果安装失败或程序接口报错，日志会给出原始报错。

## 本地运行（具备 pip 网络时）

```bash
# Raha 建议使用独立 Python 3.10 虚拟环境
python3.10 -m venv .venv-raha
source .venv-raha/bin/activate
python -m pip install raha
python optional_original_libs/run_raha_original_optional.py
deactivate

# Splink 建议使用独立 Python 3.11 虚拟环境
python3.11 -m venv .venv-splink
source .venv-splink/bin/activate
python -m pip install 'splink>=5,<6' duckdb pandas pyarrow
python optional_original_libs/run_splink5_original_optional.py
deactivate
```

## 测试口径
- Raha：虚构 `raha_dirty.csv` 与 `raha_clean.csv` 中单元格差异作真值；库自身通过真值模拟20条人工标注。属于 **有监督模拟评估**，不代表无标注检测。
- Splink：虚构 `synthetic_entities_harder.csv` 的 `record_id` 参与定位；`truth.json` 只用于评分，不输入模型。报告 pairwise Precision/Recall，但尚需添加单独的 blocking recall 和数据泄漏检查。
- **只使用合成数据！不要把真实客户数据上传到 GitHub Actions。**
- 同一批合成数据上的成绩不能外推到不同行业的真实数据，也不意味着代码可安全用于生产。
- 这两个测试只是 API 接入 smoke test + 初步计分。必须先看它们是否真正运行完成，**如果失败，不能编造结果**。
