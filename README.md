
# CIDGMed-time

This is an adapted version of the CIDGMed model (from "Causal Inference-Driven Medication Recommendation with Enhanced Dual-Granularity Learning") for step-wise medication recommendation with time regression on your custom dataset.

## Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data
```bash
python processed_step_multilabel_data_v2.py
```

### 3. Build Causal Relevance Matrices
```bash
python src/Relevance_construction_step.py
```

### 4. Train the Model
```bash
python src/train_step_cidgmed.py \
  --processed_dir processed_stepcidgmed \
  --use_scaled_time \
  --use_causal_bias
```

## Key Changes from Original CIDGMed
- **Data Format**: Uses your step-level data with clinical features, previous medications, time context
- **Relevance Matrices**: Custom Relevance_construction_step.py that uses your diagnostic-like and procedure-like features
- **Adapted Model**: StepCIDGMedFull in src/modules/CIDGMed_step.py
- **Multi-Task Training**: Medication multi-label + time regression
- **Causal Bias**: Option to use causal relevance matrices to adjust predictions
