# Predictive Modelling: dataset, pipeline and evaluation

## 1. Dataset and preprocessing
We use the Extension of the Z-Alizadeh Sani dataset (UCI): **303 patients, 55 input features** covering demographics, clinical examination, ECG, laboratory and echocardiographic measurements (34 numeric, 21 text-coded such as `Sex`, `Y/N` findings, `BBB`, `VHD`). The data has no missing values and no duplicate rows. One column (`Exertional CP`) is constant and carries no information.

Four binary targets are built from the dataset labels: **overall CAD** (from `Cath`) and stenosis in **LAD**, **LCX** and **RCA**. Prevalence is 71%, 58%, 39% and 38% respectively. Every CAD patient has at least one stenotic vessel; one patient has a stenotic vessel with a `Normal` overall label.

Preprocessing is part of each model pipeline: numeric values are median-imputed (and standardised for logistic regression), text columns are most-frequent-imputed and one-hot encoded. Because these steps are fitted inside every cross-validation fold, no information from held-out patients reaches training.

## 2. Preventing target leakage
`Cath`, `LAD`, `LCX` and `RCA` are **removed from the inputs of every model**, including when predicting a different vessel (for example, LAD status is not used to predict RCA). The training script stops with an error if any of these columns remains among the inputs.

## 3. Models and validation
Four separate classifiers are trained, one per target. For each we compared **logistic regression**, **Random Forest** (300 trees) and **gradient boosting** using **repeated stratified 5-fold cross-validation (5 repeats, 25 test folds)**, reporting accuracy, precision, recall, F1, ROC-AUC and Brier score as mean ± standard deviation over folds. The model with the highest mean ROC-AUC is kept for each target; Random Forest won all four. Its probabilities are then **calibrated** (sigmoid, internal 5-fold) because they drive the colour of each artery in the 3D view. Probabilities are mapped to bands: LOW < 0.30, MODERATE 0.30–0.60, HIGH ≥ 0.60. These cut-offs are a presentation choice, not clinically validated thresholds.

## 4. Results (cross-validated, mean ± SD)

| Target | ROC-AUC | Accuracy | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|
| Overall CAD | 0.923 ± 0.039 | 0.863 ± 0.042 | 0.883 ± 0.034 | 0.933 ± 0.038 | 0.907 ± 0.029 | 0.112 |
| LAD | 0.853 ± 0.044 | 0.790 ± 0.043 | 0.790 ± 0.045 | 0.877 ± 0.056 | 0.830 ± 0.035 | 0.164 |
| LCX | 0.732 ± 0.063 | 0.682 ± 0.053 | 0.650 ± 0.115 | 0.440 ± 0.100 | 0.516 ± 0.093 | 0.207 |
| RCA | 0.721 ± 0.054 | 0.677 ± 0.037 | 0.615 ± 0.096 | 0.398 ± 0.107 | 0.475 ± 0.088 | 0.206 |

Precision, recall and F1 use the default 0.5 decision threshold; no threshold tuning was applied. The strongest predictors found in the EDA were typical chest pain, age, Region RWMA, hypertension and diabetes (see `notebooks/EDA_sani.ipynb`).

## 5. Limitations
- **LCX and RCA are weak.** At 0.5 the models miss more than half of true stenoses (recall about 0.40–0.44). These vessels have fewer positive cases and weaker associations with the inputs, so their probabilities should be read as coarse risk indicators.
- **Small, single-source data (303 patients).** Fold-to-fold variability is large (see the SD column) and there is no external validation set.
- **Mild optimism.** The best model was chosen using the same cross-validation that is reported; nested cross-validation was not performed.
- **Referral population.** About 71% of patients have CAD because all were referred for angiography, so probabilities do not apply to a general or screening population.
- **Not a diagnostic.** Outputs are risk estimates for decision support and education only and do not replace diagnostic imaging or clinical judgement.

## 6. Interface to the rest of the system
`predict.py` exposes `predict_patient(dict)`, which returns for each of CAD, LAD, LCX and RCA a calibrated probability and a risk band. The exact inputs and allowed values are listed in `docs/input_schema.json`, with a full example in `docs/sample_patient.json`. The uncalibrated model for each target is stored alongside the calibrated one so that SHAP explanations can be computed.

## 7. Reproducing the results
`python train_cad.py` rebuilds all models and writes `reports/cv_metrics.csv`; `python make_schema.py` regenerates the input schema. A fixed random seed (42) is used throughout.
