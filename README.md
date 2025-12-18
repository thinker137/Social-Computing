## 项目简介

`scam_spam_social_computing` 是一个面向 Twitter / 社交平台的 **spam 账户检测与风险干预** 实验项目，核心目标是：

- 利用账户行为与内容统计特征，训练二分类模型识别疑似 spam 账户；
- 分析转推数 / URL 数 / @ 次数等“传播代理特征”的分布与稀疏性；
- 在高精度（Precision ≥ 0.95）约束下，给出简单可执行的阈值干预规则。

项目基于 Python 与 scikit-learn，实现了从 ARFF 数据读取、特征预处理、模型训练评估、特征消融、到干预策略评估的一整套流程。

---

## 目录结构

```text
scam_spam_social_computing/
  data/
    ICC/
      5k-continuous.arff      # 小规模连续采样子集
      5k-random.arff          # 小规模随机采样子集
      95k-continuous.arff     # 约 10 万条，连续采样（主实验）
      95k-random.arff         # 约 10 万条，随机采样（鲁棒性）
    twitter_spam.arff         # 当前使用的数据（会被 copy/覆盖）
    twitter_spam.csv          # ARFF 转成的 CSV
  outputs/                    # 最近一次运行产生的结果
  outputs_B_95k_continuous/   # 使用 95k-continuous 时的完整结果快照
  outputs_C_95k_random/       # 使用 95k-random 时的完整结果快照
  outputs_D_5k_continuous/    # 使用 5k-continuous 时的完整结果快照
  src/
    00_convert_arff_to_csv.py     # 读取 ARFF 并导出 CSV
    01_train_eval.py              # 训练 / 评估 LR & HGB 模型
    02_ablation.py                # 特征组消融实验
    03_intervention.py            # 干预阈值扫描与效果评估
    04_propagation_analysis.py    # 传播相关特征的分布与稀疏性分析
    utils.py                      # 预处理、数据划分、指标等工具函数
  README.md
  requirements.txt
  LICENSE
```

---

## 安装依赖

推荐使用 Python 3.9+。

```bash
cd scam_spam_social_computing
pip install -r requirements.txt
```

主要依赖包括：

- `numpy`、`pandas`
- `scikit-learn`、`scipy`
- `liac-arff`（读取 ARFF 数据）
- `matplotlib`（绘图）
- `joblib`（模型持久化）

---

## 数据准备

项目默认使用 NSCLab 提供的 Twitter spam ARFF 数据，已经放在 `data/ICC/` 目录中。  
你也可以替换为自己的 ARFF 文件，只要字段语义大致一致即可。

运行前通过简单的 `copy` 操作切换当前要使用的数据集，例如（Windows PowerShell / CMD）：

```bash
# 使用 95k-continuous 作为当前主数据
copy data\ICC\95k-continuous.arff data\twitter_spam.arff
```

切好之后即可进入完整流程。

---

## 运行流程（单次实验）

以当前 `data/twitter_spam.arff` 为例，建议依次运行：

```bash
python src/00_convert_arff_to_csv.py   # 读 ARFF，导出 CSV，打印列名和标签分布
python src/01_train_eval.py           # content/social/fusion + LR/HGB 训练与评估
python src/02_ablation.py             # full / minus_* 特征组消融
python src/03_intervention.py         # 阈值扫描 + 干预策略评估
python src/04_propagation_analysis.py # 传播相关特征分布与稀疏性分析
```

所有输出会写到 `outputs/` 目录中（再次运行会覆盖前一次结果）。

如果希望保留某次实验的结果，可以像本仓库中那样，将 `outputs/` 整个复制到单独目录中，例如：

```bash
Copy-Item outputs outputs_B_95k_continuous -Recurse
```

---

## 常用数据集切换示例

### 1. 快速调试：5k-random

样本量较小，适合首次跑通流程或调试：

```bash
copy data\ICC\5k-random.arff data\twitter_spam.arff
python src/00_convert_arff_to_csv.py
python src/01_train_eval.py
python src/02_ablation.py
python src/03_intervention.py
python src/04_propagation_analysis.py
```

### 2. 主实验：95k-continuous

约 10 万条样本，按数据原始顺序划分训练 / 验证 / 测试，更接近“用过去预测未来”的场景：

```bash
copy data\ICC\95k-continuous.arff data\twitter_spam.arff
python src/00_convert_arff_to_csv.py
python src/01_train_eval.py
python src/02_ablation.py
python src/03_intervention.py
python src/04_propagation_analysis.py
```

完整结果已快照到 `outputs_B_95k_continuous/`。

### 3. 鲁棒性检查：95k-random

同为 10 万条样本，但通过分层随机采样构建，用于检查模型在随机分布下的表现：

```bash
copy data\ICC\95k-random.arff data\twitter_spam.arff
python src/00_convert_arff_to_csv.py
python src/01_train_eval.py
python src/02_ablation.py
python src/03_intervention.py
python src/04_propagation_analysis.py
```

完整结果已快照到 `outputs_C_95k_random/`。

### 4. 小规模连续采样：5k-continuous

```bash
copy data\ICC\5k-continuous.arff data\twitter_spam.arff
python src/00_convert_arff_to_csv.py
python src/01_train_eval.py
python src/02_ablation.py
python src/03_intervention.py
python src/04_propagation_analysis.py
```

完整结果已快照到 `outputs_D_5k_continuous/`。

---

## 输出内容概览

不同脚本的主要输出如下（以最近一次运行的 `outputs/` 为例）：

- `outputs/01_train_eval/metrics_flat.csv`  
  - 包含 content_only / social_only / fusion_all 与 LR / HGB 在验证集和测试集上的 ROC-AUC、PR-AUC、以及在 Precision≥0.95 下的 Recall 等指标。

- `outputs/02_ablation/ablation_results.csv`  
  - 比较 full / minus_content / minus_social_account / minus_social_interaction 四种特征配置，对分析“社会信号是否有用”很方便。

- `outputs/03_intervention/intervention_threshold_scan.csv`  
  - 列出多组候选阈值下的 Precision / Recall / FPR / 干预覆盖率 / 传播代理降低比例，可用于画多种权衡曲线。

- `outputs/03_intervention/intervention_best_rule.csv`  
  - 给出在 Precision≥0.95 约束下表现最优的一条阈值规则，以及该规则对应的各项指标。

- `outputs/04_propagation/*_summary.csv`、`*_sparsity.csv`、`*.png`  
  - 对 `no_retweets` / `no_urls` / `no_usermention` 的分布、稀疏性和单特征区分能力进行统计和可视化。

---

## 注意事项

- **列名适配**  
  如果你替换了新的 ARFF 数据集，`00_convert_arff_to_csv.py` 会打印列名和类别分布。  
  如发现列名与 `src/utils.py` 中 `FEATURE_GROUPS_DEFAULT` 不一致，可以按需要修改该字典。

- **标签映射**  
  代码假设最后一列是类似 `spammer` / `non-spammer` 这样的标签，并在 `normalize_label` 中统一映射为 1/0。  
  如果你的数据标签不同（例如 spam / ham），可以在该函数中自行调整。

- **图表字体**  
  所有自动生成的图表标题与坐标轴均为英文，避免中文字体缺失导致渲染问题。  
  如需中文标题，建议在展示时在外部工具（如 PPT）中自行添加文字说明。

---

## 开源许可证

本项目采用 **MIT License** 开源许可证，详情见仓库中的 `LICENSE` 文件。

---

## 参考文献

本项目在数据与方法设计上受以下工作启发或依托：

- [1] NSCLab. (2016). Twitter spam dataset. Retrieved from NSCLab Research Group.  
- [2] Trend Micro. (2016). Web Reputation System (WRT) spam annotations.  
- [3] He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.  
- [4] Saeys, Y., Inza, I., & Larranaga, P. (2007). A review of feature selection techniques in bioinformatics. *Bioinformatics*, 23(19), 2507-2517.  
- [5] Boyd, D., Golder, S., & Lotan, G. (2010). Tweet, tweet, retweet: Conversational aspects of retweeting on Twitter. *2010 43rd Hawaii International Conference on System Sciences*, 1-10.  
- [6] Thomas, K., Grier, C., Song, D., & Paxson, V. (2011). Suspended accounts in retrospect: An analysis of Twitter spam. *Proceedings of the 2011 ACM SIGCOMM Conference on Internet Measurement*, 243-258.  
- [7] LightGBM Developers. (2023). LightGBM documentation. Retrieved from <https://lightgbm.readthedocs.io/>  
- [8] Scikit-learn Developers. (2023). Scikit-learn machine learning library. Retrieved from <https://scikit-learn.org/>  


