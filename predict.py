"""Inference helper - this is the handover point for the rest of the team.

    from predict import predict_patient
    result = predict_patient({"Age": 58, "Sex": "Male", "HTN": "Y", "LDL": 160, ...})

Any feature you leave out is imputed with the training median / most frequent value.
Unknown keys are ignored and reported back under "ignored_inputs".
"""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "models" / "cad_models.joblib"


@lru_cache(maxsize=1)
def load_bundle():
    return joblib.load(MODEL_PATH)


def _band(p, bands):
    if p < bands["low_below"]:
        return "LOW"
    return "HIGH" if p >= bands["high_from"] else "MODERATE"


def build_row(patient, bundle):
    cols = bundle["feature_columns"]
    row = pd.DataFrame([{c: patient.get(c, np.nan) for c in cols}])
    cat = bundle["categorical_columns"]
    row[cat] = row[cat].astype(object)
    return row


def predict_patient(patient: dict) -> dict:
    """Return calibrated probabilities for overall CAD and LAD / LCX / RCA stenosis."""
    bundle = load_bundle()
    row = build_row(patient, bundle)
    out = {}
    for target, info in bundle["targets"].items():
        p = float(info["calibrated"].predict_proba(row)[0, 1])
        out[target] = {"probability": round(p, 4), "risk": _band(p, bundle["risk_bands"])}
    out["missing_inputs"] = [c for c in bundle["feature_columns"] if c not in patient]
    out["ignored_inputs"] = [k for k in patient if k not in bundle["feature_columns"]]
    return out


def predict_cad(patient: dict) -> dict:
    """Overall CAD only (the function name used in the team's work division)."""
    return predict_patient(patient)["CAD"]


if __name__ == "__main__":
    import json
    print(json.dumps(predict_patient({"Age": 58, "Sex": "Male"}), indent=2))
