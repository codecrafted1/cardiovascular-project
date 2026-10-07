"""Writes docs/input_schema.json and docs/sample_patient.json from the dataset.

The schema is the contract for the patient form (Member 4): every model input, its type,
and the exact values / ranges the model was trained on.
Run:  python make_schema.py
"""
import json
from pathlib import Path

import pandas as pd

from train_cad import LABEL_COLUMNS, find_dataset, load_data

ROOT = Path(__file__).resolve().parent
df = load_data(find_dataset(None))
labels = {c for c in df.columns if c.lower() in LABEL_COLUMNS.values()}
X = df.drop(columns=list(labels))

features = []
for col in X.columns:
    s = X[col]
    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(s.unique().tolist())
        if set(vals) <= {0, 1}:
            kind, extra = "binary_0_1", {"allowed": [0, 1]}
        elif len(vals) <= 6:
            kind, extra = "ordinal", {"allowed": vals}
        else:
            kind, extra = "number", {"min": float(s.min()), "max": float(s.max()), "median": float(s.median())}
    else:
        kind, extra = "category", {"allowed": sorted(s.astype(str).unique().tolist())}
    features.append({"name": col, "type": kind, **extra})

(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "input_schema.json").write_text(json.dumps(
    {"note": "Keys must match these names exactly. Training ranges are shown; values outside them are extrapolation.",
     "features": features}, indent=2))

row = X.iloc[0].to_dict()
row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
(ROOT / "docs" / "sample_patient.json").write_text(json.dumps(row, indent=2))
print(f"Wrote docs/input_schema.json ({len(features)} features) and docs/sample_patient.json")
