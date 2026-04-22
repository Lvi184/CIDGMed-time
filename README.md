
# CIDGMed-time

## 项目目标
Step-level multi-task learning for **medication combination prediction (multilabel classification)** + **time regression** for medication sequence tasks.

## 数据预处理
运行预处理脚本，生成模型需要的数据文件：

```bash
python processed_step_multilabel_data_v2.py
```

预处理输出文件说明（保存在 `processed_stepcidgmed/`）：
- `X_multilabel.csv`: 输入特征，分为 `clinical_feature_cols`, `prev_med_cols`, `time_cols`
- `Y_multilabel_drugs.csv`: 药物多标签目标
- `y_step_time.csv`: 原始时间标签
- `y_step_time_scaled.csv`: StandardScaler 归一化后的时间标签
- `meta_columns.json`: 列分组和元信息
- `single_drug_vocab.json`: 单药词表
- `y_time_scaler.joblib`: 时间标准化 Scaler

## 训练
运行训练脚本，使用 step-level 多任务模型：

```bash
python train_step_cidgmed.py --processed_dir processed_stepcidgmed --use_scaled_time
```

## 当前范围
- ✅ 训练 drug 多标签分类 + time 回归
- ⏭️ drug.path 暂不训练（后续可扩展）
- 🔄 因果 bias correction 接口已预留，默认关闭

