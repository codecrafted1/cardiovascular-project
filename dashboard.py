
import json
from pathlib import Path

import streamlit as st
from predict import predict_patient
from explain import get_feature_importance, explain_patient

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "docs" / "input_schema.json"

st.set_page_config(
    page_title="CardioVision | Risk Explorer",
    page_icon="❤️",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background-color: #0b1020; color: #f1f5f9; }
[data-testid="stMetric"] {
    background: #151e33;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #293750;
}
.hero {
    padding: 24px;
    border-radius: 16px;
    background: linear-gradient(120deg, #172554, #312e81);
    margin-bottom: 20px;
}
.warning {
    background: #38251b;
    border: 1px solid #a16207;
    border-radius: 10px;
    padding: 14px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_schema():
    with open(SCHEMA_PATH, encoding="utf-8") as file:
        return json.load(file)["features"]

st.markdown("""
<div class="hero">
    <h1>❤️ CardioVision</h1>
    <p>Cardiovascular Risk Visualization & Prediction</p>
    <p>Explore model predictions for CAD and three coronary arteries.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Educational research prototype only. These experimental model "
    "predictions are not diagnoses and must not guide medical decisions."
)

if not (BASE_DIR / "models" / "cad_models.joblib").exists():
    st.error("Trained model not found. Run: python train_cad.py")
    st.stop()

features = load_schema()
patient = {}

st.subheader("1. Patient clinical inputs")
st.caption(
    "Enter the available clinical measurements. Use the values and "
    "categories specified by the dataset. This prototype is not a "
    "substitute for clinical assessment."
)

with st.form("patient_form"):
    for section_start in range(0, len(features), 3):
        cols = st.columns(3)

        for col, feature in zip(
            cols, features[section_start:section_start + 3]
        ):
            name = feature["name"]
            kind = feature["type"]

            with col:
                if kind == "number":
                    patient[name] = st.number_input(
                        name,
                        min_value=float(feature["min"]),
                        max_value=float(feature["max"]),
                        value=float(feature["median"]),
                        key=f"field_{name}",
                    )
                elif kind == "ordinal":
                    allowed = feature["allowed"]
                    patient[name] = st.selectbox(
                        name, allowed, key=f"field_{name}"
                    )
                elif kind == "binary_0_1":
                    patient[name] = st.selectbox(
                        name, feature["allowed"],
                        format_func=lambda value: (
                            "Yes" if value == 1 else "No"
                        ),
                        key=f"field_{name}",
                    )
                else:
                    patient[name] = st.selectbox(
                        name, feature["allowed"],
                        format_func=lambda value: (
                            "Female" if name == "Sex" and value == "Fmale"
                            else str(value)
                        ),
                        key=f"field_{name}",
                    )

    submitted = st.form_submit_button(
        "Generate predictions", type="primary", use_container_width=True
    )

if submitted:
    try:
        results = predict_patient(patient)

        st.subheader("2. Model prediction results")
        st.caption(
            "Probabilities are model outputs, not calibrated clinical "
            "risk estimates or confirmed diagnoses."
        )

        targets = ["CAD", "LAD", "LCX", "RCA"]
        columns = st.columns(4)

        labels = {
            "CAD": "Overall CAD",
            "LAD": "LAD artery",
            "LCX": "LCX artery",
            "RCA": "RCA artery",
        }

        for col, target in zip(columns, targets):
            result = results[target]
            with col:
                st.metric(
                    labels[target],
                    f"{result['probability']:.1%}",
                    help="Predicted probability from the trained model",
                )
                risk = result["risk"]
                if risk == "HIGH":
                    st.error(f"Model band: {risk}")
                elif risk == "MODERATE":
                    st.warning(f"Model band: {risk}")
                else:
                    st.success(f"Model band: {risk}")

        st.subheader("3. Prediction probability comparison")
        chart_data = {
            labels[target]: results[target]["probability"]
            for target in targets
        }
        st.bar_chart(chart_data, y_label="Predicted probability")

        st.info(
            "The LOW, MODERATE and HIGH bands use presentation cut-offs "
            "defined by the project. They are not clinically validated."
        )

        missing = results.get("missing_inputs", [])
        if missing:
            st.warning(
                f"{len(missing)} input(s) were imputed by the model."
            )

        st.caption(
            "Research prototype • Not for diagnosis or treatment decisions"
        )
        
        st.subheader("4. What influences the model?")
        st.caption(
            "These scores show overall Random Forest feature importance. "
            "They do not explain causation or prove clinical risk."
        )

        selected_target = st.selectbox(
            "Choose a prediction target",
            ["CAD", "LAD", "LCX", "RCA"],
            format_func=lambda target: {
                "CAD": "Overall CAD",
                "LAD": "LAD artery",
                "LCX": "LCX artery",
                "RCA": "RCA artery",
            }[target],
            key="importance_target",
        )

        importance = get_feature_importance(selected_target).head(10)
        importance = importance.sort_values("importance", ascending=True)

        st.bar_chart(
            importance.set_index("feature")["importance"],
            x_label="Feature",
            y_label="Importance",
            horizontal=True,
        )
        
        st.subheader("5. Why did the model make this prediction?")

        explain_target = st.selectbox(
            "Choose a target for patient-specific explanation",
            ["CAD", "LAD", "LCX", "RCA"],
            key="shap_target",
        )

        try:
            patient_explanation = explain_patient(
                patient,
                target=explain_target,
            )

            st.metric(
                "Calibrated model probability",
                f"{patient_explanation['probability']:.1%}",
            )

            shap_data = patient_explanation["explanation"].head(10).copy()
            shap_data = shap_data.sort_values("shap_value")

            st.bar_chart(
                shap_data.set_index("feature")["shap_value"],
                x_label="SHAP contribution to the explained estimator output",
                horizontal=True,
            )

        except Exception as error:
            st.error("Could not generate the patient-specific explanation.")
            st.exception(error)

    except Exception as error:
        st.error(
            "Prediction failed. Verify that model training completed "
            "successfully and that the input schema matches the model."
        )
        st.exception(error)
