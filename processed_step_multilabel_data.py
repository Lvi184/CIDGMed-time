import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# =========================
# 1. 配置
# =========================
INPUT_FILE = r"data\data_delete_variables_raw.csv"
OUTPUT_DIR = "processed_multlabel_raw"
ID_COL = "PADMNO"
LABEL_COLS = ["drug.sequence", "drug.time", "drug.path"]

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =========================
# 2. 读取数据
# =========================
df = pd.read_csv(INPUT_FILE)
print("原始数据形状:", df.shape)

required_cols = [ID_COL] + LABEL_COLS
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    raise ValueError(f"缺少关键列: {missing_required}")

# =========================
# 3. 解析函数
# =========================
def parse_step_sequence(seq):
    if pd.isna(seq):
        return []
    return [x.strip() for x in str(seq).split("-") if x.strip()]

def parse_step_times(t):
    if pd.isna(t):
        return []
    vals = []
    for x in str(t).split("+"):
        x = x.strip()
        if not x:
            continue
        vals.append(int(float(x)))
    return vals

def parse_combo_to_drugs(combo):
    if pd.isna(combo):
        return []
    return [x.strip() for x in str(combo).split("*") if x.strip()]

# =========================
# 4. 清洗普通特征
#    保留ID和标签列，不编码标签列
# =========================
df_clean = df.copy()

for col in df_clean.columns:
    if col in LABEL_COLS:
        continue

    if col == ID_COL:
        df_clean[col] = df_clean[col].astype(str)
        continue

    if pd.api.types.is_numeric_dtype(df_clean[col]):
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    else:
        df_clean[col] = df_clean[col].fillna("Unknown").astype(str)

# 编码普通类别特征，不包括ID和标签列
feature_mappings = {}
object_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()

for col in object_cols:
    if col in LABEL_COLS or col == ID_COL:
        continue
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    feature_mappings[col] = {
        str(cls): int(code)
        for cls, code in zip(le.classes_, le.transform(le.classes_))
    }

with open(Path(OUTPUT_DIR) / "feature_mappings.json", "w", encoding="utf-8") as f:
    json.dump(feature_mappings, f, ensure_ascii=False, indent=2)

# =========================
# 5. 收集全局单药词表
# =========================
all_single_drugs = set()

for _, row in df_clean.iterrows():
    step_seq = parse_step_sequence(row["drug.sequence"])
    for combo in step_seq:
        all_single_drugs.update(parse_combo_to_drugs(combo))

all_single_drugs = sorted(all_single_drugs)

with open(Path(OUTPUT_DIR) / "single_drug_vocab.json", "w", encoding="utf-8") as f:
    json.dump(all_single_drugs, f, ensure_ascii=False, indent=2)

print("单药种类数:", len(all_single_drugs))

# =========================
# 6. 构造 step-level + 单药拆分数据
# =========================
samples = []
bad_rows = []

feature_cols = [c for c in df_clean.columns if c not in LABEL_COLS]

for idx, row in df_clean.iterrows():
    patient_id = row[ID_COL]
    step_seq = parse_step_sequence(row["drug.sequence"])
    step_times = parse_step_times(row["drug.time"])

    if len(step_seq) == 0 or len(step_times) == 0:
        bad_rows.append((idx, "empty sequence/time"))
        continue

    if len(step_seq) != len(step_times):
        bad_rows.append((idx, f"length mismatch: seq={len(step_seq)}, time={len(step_times)}"))
        continue

    base_features = row[feature_cols].to_dict()
    cumulative_time = 0

    for step_idx, (combo_raw, step_time) in enumerate(zip(step_seq, step_times)):
        current_drugs = parse_combo_to_drugs(combo_raw)

        prev_combo_raw = step_seq[step_idx - 1] if step_idx > 0 else "START"
        prev_drugs = parse_combo_to_drugs(prev_combo_raw) if step_idx > 0 else []

        sample = {
            ID_COL: patient_id,
            "step": step_idx,
            "drug_combo_raw": combo_raw,
            "drug_list": "|".join(current_drugs),   # 方便看
            "step_time": step_time,
            "prev_combo_raw": prev_combo_raw,
            "prev_drug_list": "|".join(prev_drugs),
            "prev_step_time": step_times[step_idx - 1] if step_idx > 0 else 0,
            "cumulative_time_before_step": cumulative_time,
            **base_features
        }

        # 当前step的多标签
        current_drug_set = set(current_drugs)
        for drug in all_single_drugs:
            sample[f"target_{drug}"] = 1 if drug in current_drug_set else 0

        # 上一步的多标签
        prev_drug_set = set(prev_drugs)
        for drug in all_single_drugs:
            sample[f"prev_{drug}"] = 1 if drug in prev_drug_set else 0

        samples.append(sample)
        cumulative_time += step_time

step_df = pd.DataFrame(samples)

bad_rows_df = pd.DataFrame(bad_rows, columns=["row_index", "reason"])
bad_rows_df.to_csv(Path(OUTPUT_DIR) / "bad_rows.csv", index=False, encoding="utf-8-sig")

# =========================
# 7. 导出完整step数据
# =========================
step_df.to_csv(Path(OUTPUT_DIR) / "processed_step_multilabel_data.csv",
               index=False, encoding="utf-8-sig")

print("step级数据形状:", step_df.shape)

# =========================
# 8. 构造模型输入和标签
# =========================
target_cols = [f"target_{drug}" for drug in all_single_drugs]

drop_for_X = [
    ID_COL,
    "drug_combo_raw",
    "drug_list",
    "prev_combo_raw",
    "prev_drug_list",
] + target_cols

X = step_df.drop(columns=drop_for_X, errors="ignore")
Y = step_df[target_cols].copy()
y_time = step_df["step_time"].copy()
groups = step_df[ID_COL].copy()

X.to_csv(Path(OUTPUT_DIR) / "X_multilabel.csv", index=False, encoding="utf-8-sig")
Y.to_csv(Path(OUTPUT_DIR) / "Y_multilabel_drugs.csv", index=False, encoding="utf-8-sig")
y_time.to_csv(Path(OUTPUT_DIR) / "y_step_time.csv", index=False, encoding="utf-8-sig")
groups.to_csv(Path(OUTPUT_DIR) / "groups_patient_id.csv", index=False, encoding="utf-8-sig")

print("\n处理完成，输出文件：")
print("- processed_output/processed_step_multilabel_data.csv")
print("- processed_output/X_multilabel.csv")
print("- processed_output/Y_multilabel_drugs.csv")
print("- processed_output/y_step_time.csv")
print("- processed_output/groups_patient_id.csv")
print("- processed_output/single_drug_vocab.json")
print("- processed_output/feature_mappings.json")
print("- processed_output/bad_rows.csv")