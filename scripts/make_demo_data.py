import pandas as pd
from pathlib import Path


def main():
    rows = [
        {
            "PADMNO": 1,
            "gender": "F",
            "age_group": "adult",
            "admission_age": 35,
            "los": 32,
            "out_diagnosis_code": "F32.1|G47.0",
            "operation_NO": "OP01",
            "drug.sequence": "ADP*AP*OT-ADP*AP*ASE*OT-ADP*AP*OT",
            "drug.time": "9+25+24",
            "drug.path": "9xADP*AP*OT-25xADP*AP*ASE*OT-24xADP*AP*OT",
            "mood": 1,
            "bad_sleep": 1,
            "worry": 1,
        },
        {
            "PADMNO": 2,
            "gender": "M",
            "age_group": "adult",
            "admission_age": 42,
            "los": 18,
            "out_diagnosis_code": "F41.1",
            "operation_NO": "OP02",
            "drug.sequence": "AA*ADP-AA*ADP*OT-AA*ADP*AP*OT",
            "drug.time": "3+14+10",
            "drug.path": "3xAA*ADP-14xAA*ADP*OT-10xAA*ADP*AP*OT",
            "mood": 1,
            "bad_sleep": 0,
            "worry": 1,
        },
    ]
    out = Path("data/raw/raw_dataset.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved demo data to {out}")


if __name__ == "__main__":
    main()
