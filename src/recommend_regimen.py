import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from configs import data_config as cfg
from src.utils.io_utils import ensure_dir


def parse_probabilities(text: str):
    out = {}
    if not isinstance(text, str) or not text.strip():
        return out
    for part in text.split(';'):
        if ':' not in part:
            continue
        med, prob = part.rsplit(':', 1)
        try:
            out[med] = float(prob)
        except ValueError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description='Combine model medication probabilities with causal effect table to create regimen recommendations.')
    ap.add_argument('--prediction_csv', required=True)
    ap.add_argument('--causal_csv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--top_n', type=int, default=5)
    ap.add_argument('--effect_direction', choices=['reduce_los', 'increase_los'], default='reduce_los')
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    pred = pd.read_csv(args.prediction_csv)
    causal = pd.read_csv(args.causal_csv)
    if 'medication' not in causal.columns or 'ate_dr' not in causal.columns:
        raise ValueError('causal_csv must contain medication and ate_dr columns')

    effect = dict(zip(causal['medication'].astype(str), pd.to_numeric(causal['ate_dr'], errors='coerce')))
    rows = []
    for _, r in pred.iterrows():
        probs = parse_probabilities(r.get('predicted_med_probabilities', ''))
        candidates = []
        for med, prob in probs.items():
            ate = effect.get(med, np.nan)
            if np.isnan(ate):
                score = prob
            else:
                score = prob * (-ate if args.effect_direction == 'reduce_los' else ate)
            candidates.append((med, prob, ate, score))
        candidates = sorted(candidates, key=lambda x: x[3], reverse=True)[:args.top_n]
        rows.append({
            cfg.ID_COL: r.get(cfg.ID_COL, ''),
            cfg.STEP_INDEX_COL: r.get(cfg.STEP_INDEX_COL, ''),
            'recommended_meds': '|'.join([x[0] for x in candidates]),
            'recommendation_detail': ';'.join([f'{x[0]}:prob={x[1]:.4f},ate={x[2]:.4f},score={x[3]:.4f}' for x in candidates]),
            'predicted_los': r.get('predicted_los', np.nan),
            'predicted_step_duration': r.get('predicted_step_duration', np.nan),
            'true_med_regimen': r.get('true_med_regimen', ''),
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / 'regimen_recommendations.csv', index=False, encoding='utf-8-sig')
    print(f'Saved recommendations to {out_dir / "regimen_recommendations.csv"}')


if __name__ == '__main__':
    main()
