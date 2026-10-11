
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

