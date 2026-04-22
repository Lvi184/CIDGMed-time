
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def compute_causal_effects_simple(data_dir='processed_stepcidgmed'):
    """
    Simple approach to compute causal-like effects:
    - For each diagnostic/procedure feature, compute its effect on each medication
    - Using logistic regression (adjusting for confounders like previous medication)
    """
    
    data_dir = Path(data_dir)
    
    # Load data
    X = pd.read_csv(data_dir / 'X_multilabel.csv')
    Y_med = pd.read_csv(data_dir / 'Y_multilabel_drugs.csv')
    meta = json.load(open(data_dir / 'meta_columns.json', 'r'))
    single_drugs = json.load(open(data_dir / 'single_drug_vocab.json', 'r'))
    
    # Define features for causal effect computation
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
    
    confounders = meta['prev_med_cols']  # use previous medication as confounders
    
    # Initialize effect matrix
    num_features = len(diagnostic_like_features) + len(procedure_like_features)
    num_drugs = len(single_drugs)
    effect_matrix = np.zeros((num_features, num_drugs))
    
    all_features = diagnostic_like_features + procedure_like_features
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = X.copy()
    for col in all_features:
        if col in X.columns:
            X_scaled[col] = scaler.fit_transform(X_scaled[[col]].fillna(0))
            
    # Compute effects
    for feat_idx, feat in enumerate(all_features):
        if feat not in X.columns:
            continue
            
        # Prepare features for regression
        reg_features = [feat] + confounders
        X_reg = X_scaled[reg_features].fillna(0).values
        
        for med_idx, med in enumerate(single_drugs):
            y = Y_med.iloc[:, med_idx].values
            
            # Skip if too few positive examples
            if np.sum(y) < 2:
                continue
            
            try:
                # Fit logistic regression
                model = LogisticRegression(max_iter=1000, C=1.0)
                model.fit(X_reg, y)
                
                # Extract coefficient for feature of interest
                coef = model.coef_[0][0]
                effect_matrix[feat_idx, med_idx] = coef
            except Exception as e:
                print(f"Error for {feat} and {med}: {e}")
                continue
    
    # Save effect matrix
    feature_names = diagnostic_like_features + procedure_like_features
    np.save(data_dir / 'causal_effect_matrix.npy', effect_matrix)
    json.dump(feature_names, open(data_dir / 'causal_feature_names.json', 'w'), ensure_ascii=False, indent=2)
    print("Causal effect matrix saved!")
    
    # Print top effects for each drug
    print("\nTop causal effects per medication:")
    for med_idx, med in enumerate(single_drugs):
        # Get top 3 effects for this drug
        top_indices = np.argsort(np.abs(effect_matrix[:, med_idx]))[-3:][::-1]
        print(f"\n{med}:")
        for idx in top_indices:
            if abs(effect_matrix[idx, med_idx]) > 0.01:
                print(f"  {feature_names[idx]}: {effect_matrix[idx, med_idx]:.4f}")


if __name__ == "__main__":
    compute_causal_effects_simple()
