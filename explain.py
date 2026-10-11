
"""Feature importance for the trained CardioVision Random Forest models."""

from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cad_models.joblib"


def get_feature_importance(target="CAD"):
    """Return ranked feature importance for one trained target model."""
    bundle = joblib.load(MODEL_PATH)

    if target not in bundle["targets"]:
        available = ", ".join(bundle["targets"].keys())
        raise ValueError(f"Unknown target {target!r}. Choose from: {available}")

    info = bundle["targets"][target]
    model = info["base"]

    # Access the fitted preprocessing and classifier steps.
    preprocessor = model.named_steps["pre"]
    classifier = model.named_steps["clf"]

    if not hasattr(classifier, "feature_importances_"):
        raise ValueError(
            f"The {target} classifier does not expose tree feature importance."
        )

    # Convert encoded feature names back to readable labels where possible.
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError("Feature names do not match the model importance values.")

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    # Clean common ColumnTransformer prefixes for display.
    result["feature"] = (
        result["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    return result.sort_values("importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    for target in ("CAD", "LAD", "LCX", "RCA"):
        print(f"\nTop 10 features for {target}")
        print(get_feature_importance(target).head(10).to_string(index=False))


def explain_patient(patient: dict, target="CAD"):
    """Explain one patient's prediction using SHAP.

    Returns the calibrated model probability and approximate feature
    contributions from one fitted Random Forest estimator.
    """
    import shap
    import numpy as np

    from predict import load_bundle, build_row

    bundle = load_bundle()

    if target not in bundle["targets"]:
        raise ValueError(f"Unknown target: {target}")

    row = build_row(patient, bundle)
    calibrated = bundle["targets"][target]["calibrated"]

    # Use the first fitted estimator's preprocessing and Random Forest.
    pipeline = calibrated.calibrated_classifiers_[0].estimator
    preprocessor = pipeline.named_steps["pre"]
    classifier = pipeline.named_steps["clf"]

    transformed = preprocessor.transform(row)
    feature_names = preprocessor.get_feature_names_out()

    # SHAP output for binary classification: (samples, features, classes).
    explainer = shap.TreeExplainer(classifier)
    values = np.asarray(explainer.shap_values(transformed))

    if values.ndim == 3:
        # Class 1 represents the positive class.
        contributions = values[0, :, 1]
    elif values.ndim == 2:
        contributions = values[0]
    else:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

    # Clean preprocessing prefixes for readable feature names.
    names = [
        name.replace("num__", "").replace("cat__", "")
        for name in feature_names
    ]

    explanation = pd.DataFrame({
        "feature": names,
        "shap_value": contributions,
    })
    explanation["abs_shap"] = explanation["shap_value"].abs()
    explanation = explanation.sort_values(
        "abs_shap", ascending=False
    ).reset_index(drop=True)

    return {
        "probability": calibrated.predict_proba(row)[0, 1],
        "explanation": explanation,
        "method": "First fitted Random Forest estimator",
    }
