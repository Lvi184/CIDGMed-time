# CIDGMed Baseline-to-Step Causal + Prediction Project

这是一个针对当前数据结构的完整项目：把住院级用药路径展开为 step-level 长表，进行近似因果效应分析，并联合预测：

1. 最终时间结局：默认 `los`
2. 阶段时长：来自 `drug.time` 展开的 `step_duration`
3. 阶段药物方案：多标签药物组合 `target_med_codes`
4. 推荐药物候选：结合模型预测概率与因果效应表生成 `regimen_recommendations.csv`

> 重要说明：由于数据只有初始基线特征，没有每个 step 更新后的状态变量，本项目的因果结果应解释为“在基线特征和既往用药历史调整下的 step-level 近似因果效应”，不是严格动态治疗策略因果识别。

---

## 目录结构

```text
cidgmed_full_project/
├── configs/
│   └── data_config.py
├── src/
│   ├── data_builder_step.py
│   ├── feature_builder_step.py
│   ├── causal_effect_builder_step.py
│   ├── train_step_cidgmed.py
│   ├── recommend_regimen.py
│   ├── models/step_cidgmed.py
│   └── utils/
├── scripts/
│   ├── make_demo_data.py
│   └── run_pipeline.sh
├── data/raw/
├── data/processed/
├── requirements.txt
└── README.md
```

---

## 输入字段

- `PADMNO`：患者/住院记录 ID
- `drug.sequence`：阶段路径，例如 `ADP*AP*OT-ADP*AP*ASE*OT`
- `drug.time`：阶段时长，例如 `9+25+24`
- `drug.path`：带时长前缀的路径，例如 `9xADP*AP*OT-25xADP*AP*ASE*OT`
- `los`：默认最终时间结局
- `out_diagnosis_code`、`operation_NO`：诊断与操作编码
- 其他人口学、症状、检查、检验、共病、病史字段见 `configs/data_config.py`

---

## 一键运行

把你的原始 CSV 放到：

```text
data/raw/raw_dataset.csv
```

安装依赖并运行：

```bash
pip install -r requirements.txt
bash scripts/run_pipeline.sh data/raw/raw_dataset.csv data/processed 20
```

也可以逐步运行：

```bash
python -m src.data_builder_step --input data/raw/raw_dataset.csv --output data/processed/visit_level_step.csv
python -m src.feature_builder_step --input data/processed/visit_level_step.csv --out_dir data/processed
python -m src.causal_effect_builder_step --data_csv data/processed/visit_level_step.csv --feature_dir data/processed --out_dir data/processed
python -m src.train_step_cidgmed --data_csv data/processed/visit_level_step.csv --feature_dir data/processed --out_dir data/processed --epochs 50 --batch_size 32
python -m src.recommend_regimen --prediction_csv data/processed/prediction_preview.csv --causal_csv data/processed/single_med_effect_on_los.csv --out_dir data/processed
```

---

## 核心输出

### 数据展开

- `visit_level_step.csv`

关键列：

- `PADMNO`
- `step_idx`
- `step_duration`
- `target_med_codes`
- `prev_med_codes`
- `added_med_codes`
- `removed_med_codes`
- `diag_codes`
- `proc_codes`
- `los`

### 特征矩阵

- `X_diag.npy`
- `X_proc.npy`
- `X_prev_med.npy`
- `X_demo.npy`
- `X_confounders.npy`
- `Y_med.npy`
- `Y_time.npy`
- `Y_step_duration.npy`
- `vocab.pkl`

### 因果分析

- `single_med_effect_on_los.csv`：单药对 `los` 的近似 doubly robust ATE
- `step_duration_effect_on_los.csv`：阶段时长与 outcome 的描述统计
- `regimen_frequency.csv`：方案频率

### 模型预测

- `best_model.pt`
- `train_metrics.json`
- `prediction_preview.csv`
- `predicted_med_prob.npy`

`prediction_preview.csv` 现在包含：

- `predicted_los`：预测最终住院时长/结局时间
- `true_los`
- `predicted_step_duration`：预测阶段时长
- `true_step_duration`
- `predicted_med_regimen`：预测药物组合
- `predicted_med_probabilities`：候选药物概率
- `true_med_regimen`

### 药物推荐

- `regimen_recommendations.csv`

该文件会把模型预测的药物概率与 `single_med_effect_on_los.csv` 中的 `ate_dr` 结合，默认优先推荐“预测概率较高且关联更短 LOS”的药物候选。它是辅助分析结果，不能直接替代临床决策。

---

## 模型说明

`StepCIDGMedNet` 是多任务神经网络：

- shared backbone：输入基线特征、诊断、操作、既往用药历史、step 信息
- `med_head`：多标签预测当前/下一阶段药物组合
- `time_head`：预测最终 outcome，默认 `los`
- `duration_head`：预测阶段时长 `step_duration`

训练损失：

```text
loss = BCE(med) + MSE(los) + duration_weight * MSE(step_duration)
```

可调参数：

```bash
python -m src.train_step_cidgmed \
  --data_csv data/processed/visit_level_step.csv \
  --feature_dir data/processed \
  --out_dir data/processed \
  --epochs 50 \
  --batch_size 32 \
  --duration_weight 0.5 \
  --med_threshold 0.5 \
  --top_k_meds 5
```

---

## 论文/报告建议表述

推荐表述：

> 本研究基于住院初始基线特征、诊断/操作编码及既往给药历史，构建 step-level 用药路径数据集，采用倾向评分、逆概率加权与 doubly robust 估计，分析阶段性药物暴露与最终住院时长之间的调整后效应；同时训练多任务神经网络联合预测药物方案、阶段时长及最终时间结局。

不建议表述为：

> 严格动态治疗策略因果推断。

原因是当前数据缺少每个 step 后更新的病情状态、实验室指标或症状变化。

---

## 评价指标分析模块

本版本新增 `src/evaluate_project.py`，用于在训练和推荐结束后自动评估模型效果。`scripts/run_pipeline.sh` 已经把评估步骤接入完整流水线，默认评估验证集 `--subset val`。

### 一键生成评价结果

```bash
bash scripts/run_pipeline.sh data/raw/raw_dataset.csv data/processed 20
```

运行完成后会额外生成：

```text
data/processed/evaluation_metrics.json
data/processed/evaluation_summary.csv
data/processed/per_medication_metrics_val.csv
data/processed/causal_diagnostics.csv
data/processed/evaluation_plots/
```

### 单独运行评估

```bash
python -m src.evaluate_project --data_csv data/processed/visit_level_step.csv --feature_dir data/processed --prediction_csv data/processed/prediction_preview.csv --causal_csv data/processed/single_med_effect_on_los.csv --out_dir data/processed --subset val --med_threshold 0.5 --top_k 3,5,10 --causal_top_n 10 --bootstrap 100

python -m src.evaluate_project \
  --data_csv data/processed/visit_level_step.csv \
  --feature_dir data/processed \
  --prediction_csv data/processed/prediction_preview.csv \
  --causal_csv data/processed/single_med_effect_on_los.csv \
  --out_dir data/processed \
  --subset val \
  --med_threshold 0.5 \
  --top_k 3,5,10 \
  --causal_top_n 10 \
  --bootstrap 100
```

`--subset` 可选：

- `val`：验证集，推荐用于报告模型泛化效果
- `train`：训练集，用于检查是否过拟合
- `all`：全量数据，用于整体描述

### 已加入的指标

#### 1. 药物方案预测，多标签分类

输出在 `evaluation_metrics.json`、`evaluation_summary.csv` 和 `per_medication_metrics_val.csv` 中。

核心指标：

- `micro_f1`
- `macro_f1`
- `samples_f1`
- `micro_precision`
- `micro_recall`
- `jaccard_samples`
- `hamming_loss`
- `subset_exact_match`
- `recall_at_3`
- `recall_at_5`
- `recall_at_10`
- `precision_at_3`
- `precision_at_5`
- `precision_at_10`
- `hit_rate_at_3`
- `hit_rate_at_5`
- `hit_rate_at_10`
- `micro_roc_auc`
- `micro_average_precision`
- `label_ranking_average_precision`

推荐报告优先展示：

```text
Micro-F1, Macro-F1, Jaccard, Recall@5, Precision@5
```

其中：

- `Micro-F1` 看整体药物预测能力
- `Macro-F1` 看低频药物是否也能预测
- `Jaccard` 看整套药物组合与真实组合的重叠度
- `Recall@K` 看真实药物是否进入前 K 个推荐候选

#### 2. LOS 和阶段时长预测，回归任务

分别输出：

- `los_regression`
- `step_duration_regression`

核心指标：

- `mae`
- `median_absolute_error`
- `rmse`
- `r2`
- `mape`
- `bias_mean_pred_minus_true`

推荐报告优先展示：

```text
MAE, RMSE, R²
```

#### 3. 因果分析诊断

输出在：

```text
data/processed/causal_diagnostics.csv
```

包含：

- `ate_dr`
- `ate_ci95_low`
- `ate_ci95_high`
- `treated_n`
- `control_n`
- `propensity_min / p05 / p50 / p95 / max`
- `ess_total`
- `ess_treated`
- `ess_control`
- `mean_abs_smd_before`
- `mean_abs_smd_after`
- `max_abs_smd_before`
- `max_abs_smd_after`
- `pct_abs_smd_lt_0_1_before`
- `pct_abs_smd_lt_0_1_after`

推荐报告优先展示：

```text
ATE, 95% CI, SMD before/after, ESS
```

解释建议：

- `mean_abs_smd_after` 越低越好，通常希望小于 0.1
- `ESS` 太小说明 IPW 权重不稳定，因果估计可信度下降
- `ATE 95% CI` 如果跨 0，说明该药物对 LOS 的方向不够稳定

### 自动图表

图表输出在：

```text
data/processed/evaluation_plots/
```

包括：

- `los_pred_vs_true.png`
- `step_duration_pred_vs_true.png`
- `med_probability_histogram.png`
- `causal_smd_balance.png`

这些图可以直接放到论文、答辩 PPT 或项目报告中。
