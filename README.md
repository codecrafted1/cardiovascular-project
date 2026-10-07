# Cardiovascular Risk Visualization & Prediction (Multimodal AI Hackathon 2026, Track A)

Predicts **overall coronary artery disease (CAD)** and **stenosis of the three main coronary arteries (LAD, LCX, RCA)** from clinical, ECG, laboratory and echocardiographic data. The predictions are meant to drive an interactive 3D heart view and an explainability dashboard.

> **Disclaimer:** predictions are for decision support and educational purposes only. They are not a substitute for formal diagnostic imaging or professional medical judgement.

## Project status

| Part | Status |
|---|---|
| Prediction pipeline (CAD + LAD / LCX / RCA) | Done: `train_cad.py`, `predict.py` |
| Exploratory data analysis | Done: `notebooks/EDA_sani.ipynb` |
| Input specification for the patient form | Done: `docs/input_schema.json` |
| SHAP explanations | In progress |
| 3D heart / coronary visualization | In progress |
| Web dashboard and API | In progress |

## Dataset

Extension of the Z-Alizadeh Sani dataset (UCI Machine Learning Repository): 303 patients, 55 input features, four labels. The file is in `datasets/z_alizadeh_sani_extension.xlsx`.

`Cath` (overall CAD), `LAD`, `LCX` and `RCA` are **targets only**. They are removed from the inputs of every model to prevent target leakage.

## Results (repeated stratified 5-fold cross-validation, mean ± SD)

| Target | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Overall CAD | 0.923 ± 0.039 | 0.863 ± 0.042 | 0.883 ± 0.034 | 0.933 ± 0.038 | 0.907 ± 0.029 |
| LAD | 0.853 ± 0.044 | 0.790 ± 0.043 | 0.790 ± 0.045 | 0.877 ± 0.056 | 0.830 ± 0.035 |
| LCX | 0.732 ± 0.063 | 0.682 ± 0.053 | 0.650 ± 0.115 | 0.440 ± 0.100 | 0.516 ± 0.093 |
| RCA | 0.721 ± 0.054 | 0.677 ± 0.037 | 0.615 ± 0.096 | 0.398 ± 0.107 | 0.475 ± 0.088 |

Random Forest was the best of three compared models (logistic regression, Random Forest, gradient boosting) for all four targets. Precision, recall and F1 use the default 0.5 threshold. LCX and RCA are noticeably weaker: they have fewer positive cases, and the models miss more than half of the true stenoses. Full numbers for every model are in `reports/cv_metrics.csv`; the method and limitations are described in `docs/member1_documentation.md`.

## Setup

```bash
git clone https://github.com/triiishaha/cardiovascular-project.git
cd cardiovascular-project
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Usage

**1. Train the models** (about 2 minutes). Writes `models/cad_models.joblib`, `models/metrics.json` and `reports/cv_metrics.csv`:

```bash
python train_cad.py
```

**2. Predict for a patient:**

```python
import json
from predict import predict_patient

patient = json.load(open("docs/sample_patient.json"))
print(predict_patient(patient))
# {'CAD': {'probability': 0.87, 'risk': 'HIGH'}, 'LAD': {...}, 'LCX': {...}, 'RCA': {...}, ...}
```

Risk bands: LOW below 0.30, MODERATE 0.30 to 0.60, HIGH from 0.60. These are presentation cut-offs, not clinically validated thresholds.

**3. Regenerate the input specification** (after any dataset change):

```bash
python make_schema.py
```

`docs/input_schema.json` lists every model input with its type and allowed values or range. Requests must use these exact names and values (for example `Sex` is `Male` or `Fmale`). Missing fields are filled with training-set typical values, so the patient form should require all fields.

## Project structure

```
train_cad.py            Train and evaluate the CAD / LAD / LCX / RCA models
predict.py              predict_patient(dict) -> probability and risk band per target
make_schema.py          Generates docs/input_schema.json and docs/sample_patient.json
datasets/               Z-Alizadeh Sani extension dataset (and legacy cardio_train.csv)
docs/                   Input schema, sample patient, modelling documentation
notebooks/EDA_sani.ipynb  Exploratory analysis of the Track A dataset
reports/cv_metrics.csv  Cross-validation results for every model and target
models/                 Trained models (created by train_cad.py, not stored in git)
app.py, train.py        Earlier baseline (general cardiovascular dataset), kept for reference
```

## Notes

- Trained model files are not stored in git (`.gitignore`). Run `python train_cad.py` after cloning.
- The earlier baseline (`app.py`, `train.py`, `notebooks/EDA.ipynb`) uses the Kaggle cardiovascular dataset and a different task; it is separate from the Track A pipeline.
- Dataset source: UCI Machine Learning Repository, Extension of the Z-Alizadeh Sani dataset.
