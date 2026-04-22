
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Load processed step data
processed_dir = Path('processed_stepcidgmed')
X = pd.read_csv(processed_dir / 'X_multilabel.csv')
Y_med = pd.read_csv(processed_dir / 'Y_multilabel_drugs.csv')
meta = json.load(open(processed_dir / 'meta_columns.json', 'r'))
groups = pd.read_csv(processed_dir / 'groups_patient_id.csv', header=None).values.flatten()

# Define diagnostic-like and procedure-like feature groups (from your data)
diagnostic_like_features = [
    'out_diagnosis_code',
    'severity', 
    'first_episode',
    'psychiatric_comorbidity',
    'endocrine_comorbidity',
    'nervous_comorbidity',
    'digestive_comorbidity',
    'circulatory_comorbidity',
    'respiratory_comorbidity',
    'cancer_comorbidity',
]

procedure_like_features = [
    'surgery_NO',
    'operation_NO',
    'history_surgery',
]

# Get vocab sizes for diagnostic-like, procedure-like, and meds
num_diag = len(diagnostic_like_features)
num_proc = len(procedure_like_features)
num_med = len(meta['target_cols'])

# Initialize effect matrices
diag_med_matrix = np.zeros((num_diag, num_med))
proc_med_matrix = np.zeros((num_proc, num_med))

diag_count = np.zeros(num_diag)
proc_count = np.zeros(num_proc)

# Populate matrices by step (simple correlation for baseline)
for idx in range(len(X)):
    # Get current med labels
    current_meds = Y_med.iloc[idx].values
    
    # Add to diag-med matrix
    for diag_idx, diag_feat in enumerate(diagnostic_like_features):
        if diag_feat in X.columns:
            diag_val = X[diag_feat].iloc[idx]
            if pd.notna(diag_val) and diag_val > 0:  # present or elevated
                diag_med_matrix[diag_idx] += current_meds
                diag_count[diag_idx] += 1
                
    # Add to proc-med matrix
    for proc_idx, proc_feat in enumerate(procedure_like_features):
        if proc_feat in X.columns:
            proc_val = X[proc_feat].iloc[idx]
            if pd.notna(proc_val) and proc_val > 0:  # present or elevated
                proc_med_matrix[proc_idx] += current_meds
                proc_count[proc_idx] += 1

# Normalize by count (avoid division by zero)
for i in range(num_diag):
    if diag_count[i] > 0:
        diag_med_matrix[i] /= diag_count[i]

for i in range(num_proc):
    if proc_count[i] > 0:
        proc_med_matrix[i] /= proc_count[i]

# Save matrices
np.save(processed_dir / 'Diag_Med_relevance.npy', diag_med_matrix)
np.save(processed_dir / 'Proc_Med_relevance.npy', proc_med_matrix)
print('Relevance matrices saved!')
