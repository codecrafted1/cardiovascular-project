"""Track A - ML pipeline: overall CAD + LAD / LCX / RCA stenosis classifiers.

Usage:
    python train_cad.py                      # auto-finds datasets/*sani* file
    python train_cad.py --data path/to/file  # .xlsx or .csv

What it does
  1. Loads the Z-Alizadeh Sani (extension) dataset.
  2. Builds four binary targets: CAD (from `Cath`), LAD, LCX, RCA.
  3. Removes Cath, LAD, LCX and RCA from the inputs for EVERY model (no target leakage).
  4. Compares Logistic Regression, Random Forest and Gradient Boosting with repeated
     stratified 5-fold CV. Imputation / encoding / scaling happen inside each fold.
  5. Picks the best model per target by mean ROC-AUC, calibrates its probabilities,
     and saves everything to models/cad_models.joblib.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# target name -> label column (case-insensitive) in the dataset
LABEL_COLUMNS = {"CAD": "cath", "LAD": "lad", "LCX": "lcx", "RCA": "rca"}
# Team convention for turning a probability into a band. Change here if the team agrees otherwise.
RISK_BANDS = {"low_below": 0.30, "high_from": 0.60}

POSITIVE = {"cad", "stenotic", "stenosis", "yes", "y", "1", "1.0", "true", "positive", "abnormal"}
NEGATIVE = {"normal", "no", "n", "0", "0.0", "false", "negative"}


# ----------------------------------------------------------------------------- data
def find_dataset(explicit):
    if explicit:
        return Path(explicit)
    folder = ROOT / "datasets"
    hits = sorted(
        p for p in folder.glob("*")
        if p.suffix.lower() in {".xlsx", ".xls", ".csv"} and ("sani" in p.name.lower() or "alizadeh" in p.name.lower())
    )
    if not hits:
        sys.exit(
            "Dataset not found. Put the Z-Alizadeh Sani file in datasets/ "
            "(name should contain 'sani' or 'alizadeh') or pass --data <path>."
        )
    return hits[0]


def load_data(path):
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def to_binary(series, name):
    if series.isna().any():
        sys.exit(f"Label column '{name}' has missing values; clean the file first.")
    if pd.api.types.is_numeric_dtype(series):
        if not set(series.unique()) <= {0, 1}:
            sys.exit(f"Label column '{name}' is numeric but not 0/1: {sorted(series.unique())[:10]}")
        return series.astype(int)
    mapped = series.astype(str).str.strip().str.lower().map(
        lambda v: 1 if v in POSITIVE else (0 if v in NEGATIVE else np.nan)
    )
    if mapped.isna().any():
        bad = sorted(set(series[mapped.isna()].astype(str)))
        sys.exit(f"Don't know how to read values {bad} in label column '{name}'. "
                 f"Add them to POSITIVE/NEGATIVE in train_cad.py.")
    return mapped.astype(int)


def split_features_targets(df):
    lower = {c.lower(): c for c in df.columns}
    missing = [v for v in LABEL_COLUMNS.values() if v not in lower]
    if missing:
        sys.exit(f"Missing label columns {missing}. Columns found: {list(df.columns)}")
    label_cols = {t: lower[v] for t, v in LABEL_COLUMNS.items()}
    y = {t: to_binary(df[c], c) for t, c in label_cols.items()}
    X = df.drop(columns=list(label_cols.values()))
    # Hard leakage guard: none of the label columns may remain as an input.
    assert not (set(LABEL_COLUMNS.values()) & {c.lower() for c in X.columns}), "target leakage!"
    return X, y, label_cols


# ----------------------------------------------------------------------------- models
def build_pipeline(kind, num_cols, cat_cols):
    scale = kind == "logreg"
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        sparse_threshold=0,
    )
    if kind == "logreg":
        clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
    elif kind == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                     random_state=SEED, n_jobs=1)
    elif kind == "grad_boost":
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                             l2_regularization=1.0, random_state=SEED)
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("clf", clf)])


MODEL_KINDS = ["logreg", "random_forest", "grad_boost"]
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "brier": "neg_brier_score",
}


def band(p):
    if p < RISK_BANDS["low_below"]:
        return "LOW"
    return "HIGH" if p >= RISK_BANDS["high_from"] else "MODERATE"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to the dataset (.xlsx or .csv)")
    ap.add_argument("--repeats", type=int, default=5, help="CV repeats (default 5 x 5 folds)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    data_path = find_dataset(args.data)
    df = load_data(data_path)
    X, y, label_cols = split_features_targets(df)
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    X[cat_cols] = X[cat_cols].astype(object)

    print(f"Dataset: {data_path.name}  rows={len(df)}  input features={X.shape[1]} "
          f"({len(num_cols)} numeric, {len(cat_cols)} categorical)")
    print(f"Excluded from inputs (leakage): {sorted(label_cols.values())}")
    print("\nClass balance (1 = disease / stenosis):")
    for t, s in y.items():
        pos = int(s.sum())
        flag = "   <-- very few positives, treat metrics with caution" if min(pos, len(s) - pos) < 20 else ""
        print(f"  {t:<4} positives={pos:<4} negatives={len(s) - pos:<4} prevalence={pos / len(s):.1%}{flag}")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)
    rows, bundle_targets = [], {}

    for target, labels in y.items():
        print(f"\n=== {target} ===")
        for kind in MODEL_KINDS:
            res = cross_validate(build_pipeline(kind, num_cols, cat_cols), X, labels,
                                 cv=cv, scoring=SCORING, n_jobs=-1)
            row = {"target": target, "model": kind}
            for m in SCORING:
                vals = res[f"test_{m}"] * (-1 if m == "brier" else 1)
                row[f"{m}_mean"], row[f"{m}_std"] = vals.mean(), vals.std()
            rows.append(row)
            print(f"  {kind:<14} AUC {row['roc_auc_mean']:.3f}±{row['roc_auc_std']:.3f}  "
                  f"acc {row['accuracy_mean']:.3f}  prec {row['precision_mean']:.3f}  "
                  f"rec {row['recall_mean']:.3f}  f1 {row['f1_mean']:.3f}  brier {row['brier_mean']:.3f}")

        best = max((r for r in rows if r["target"] == target), key=lambda r: r["roc_auc_mean"])
        print(f"  -> best by ROC-AUC: {best['model']}")

        base = build_pipeline(best["model"], num_cols, cat_cols).fit(X, labels)
        n_min = int(min(labels.sum(), len(labels) - labels.sum()))
        calibrated = CalibratedClassifierCV(
            clone(build_pipeline(best["model"], num_cols, cat_cols)),
            method="sigmoid", cv=max(2, min(5, n_min)),
        ).fit(X, labels)
        bundle_targets[target] = {"model_name": best["model"], "base": base, "calibrated": calibrated}

    metrics = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    metrics.to_csv(REPORTS_DIR / "cv_metrics.csv", index=False)

    best_summary = {}
    for t, info in bundle_targets.items():
        r = metrics[(metrics.target == t) & (metrics.model == info["model_name"])].iloc[0]
        best_summary[t] = {"model": info["model_name"],
                           **{k: round(float(r[k]), 4) for k in r.index if k.endswith(("_mean", "_std"))}}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(best_summary, indent=2))

    joblib.dump(
        {
            "feature_columns": list(X.columns),
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "excluded_label_columns": sorted(label_cols.values()),
            "risk_bands": RISK_BANDS,
            "n_training_rows": len(df),
            "targets": bundle_targets,
        },
        MODELS_DIR / "cad_models.joblib",
    )
    print(f"\nSaved: models/cad_models.joblib, models/metrics.json, reports/cv_metrics.csv")


if __name__ == "__main__":
    main()
