#!/usr/bin/env bash
set -euo pipefail
RAW_CSV=${1:-data/raw/raw_dataset.csv}
OUT_DIR=${2:-data/processed}
EPOCHS=${3:-20}

python -m src.data_builder_step --input "$RAW_CSV" --output "$OUT_DIR/visit_level_step.csv"
python -m src.feature_builder_step --input "$OUT_DIR/visit_level_step.csv" --out_dir "$OUT_DIR"
python -m src.causal_effect_builder_step --data_csv "$OUT_DIR/visit_level_step.csv" --feature_dir "$OUT_DIR" --out_dir "$OUT_DIR"
python -m src.train_step_cidgmed --data_csv "$OUT_DIR/visit_level_step.csv" --feature_dir "$OUT_DIR" --out_dir "$OUT_DIR" --epochs "$EPOCHS" --batch_size 32
python -m src.recommend_regimen --prediction_csv "$OUT_DIR/prediction_preview.csv" --causal_csv "$OUT_DIR/single_med_effect_on_los.csv" --out_dir "$OUT_DIR"
python -m src.evaluate_project --data_csv "$OUT_DIR/visit_level_step.csv" --feature_dir "$OUT_DIR" --prediction_csv "$OUT_DIR/prediction_preview.csv" --causal_csv "$OUT_DIR/single_med_effect_on_los.csv" --out_dir "$OUT_DIR" --subset val --med_threshold 0.5 --top_k 3,5,10 --causal_top_n 10 --bootstrap 100
