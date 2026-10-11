
import json
from pathlib import Path

import pytest

from predict import predict_patient
from explain import explain_patient, get_feature_importance

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "docs" / "sample_patient.json"
MODEL_PATH = ROOT / "models" / "cad_models.joblib"

TARGETS = ["CAD", "LAD", "LCX", "RCA"]


@pytest.fixture
def patient():
    if not SAMPLE_PATH.exists():
        pytest.skip("Sample patient file is unavailable.")
    with SAMPLE_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(autouse=True)
def require_trained_model():
    if not MODEL_PATH.exists():
        pytest.skip("Run python train_cad.py first.")


def test_predictions_cover_all_targets(patient):
    results = predict_patient(patient)

    for target in TARGETS:
        assert target in results
        assert 0.0 <= results[target]["probability"] <= 1.0
        assert results[target]["risk"] in {
            "LOW", "MODERATE", "HIGH"
        }


def test_missing_and_unknown_inputs_are_reported(patient):
    results = predict_patient({
        "Age": patient.get("Age", 58),
        "unexpected_test_field": 123,
    })

    assert isinstance(results["missing_inputs"], list)
    assert "unexpected_test_field" in results["ignored_inputs"]


@pytest.mark.parametrize("target", TARGETS)
def test_global_feature_importance(target):
    result = get_feature_importance(target)

    assert not result.empty
    assert {"feature", "importance"}.issubset(result.columns)
    assert result["importance"].notna().all()


@pytest.mark.parametrize("target", TARGETS)
def test_patient_shap_explanation(target, patient):
    result = explain_patient(patient, target=target)

    assert 0.0 <= result["probability"] <= 1.0
    assert not result["explanation"].empty
    assert {
        "feature", "shap_value", "abs_shap"
    }.issubset(result["explanation"].columns)
    assert result["explanation"]["shap_value"].notna().all()
