import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/heart_model.pkl")

st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="centered"
)
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

div[data-testid="stMetric"] {
    background-color:#262730;
    border-radius:10px;
    padding:15px;
}

.stButton>button {
    width:100%;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stAlert {
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

st.title("❤️ AI-Based Cardiovascular Disease Risk Prediction")
 
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=80)

    st.title("About")

    st.info(
        """
        **AI-Based Cardiovascular Disease Risk Prediction**

        This application predicts the likelihood of cardiovascular disease
        using a Machine Learning model.

        Developed using:
        - Streamlit
        - Scikit-learn
        - Pandas
        """
    )

   
   
   
    st.markdown("""
Predict the likelihood of cardiovascular disease using a Machine Learning model.

Fill in the patient's health information below and click **Predict**.
""")




col1, col2 = st.columns(2)

with col1:
    age_years = st.number_input(
        "Age (years)", min_value=1, max_value=120, value=40
    )

    height = st.number_input(
        "Height (cm)", min_value=100, max_value=230, value=165
    )

    ap_hi = st.number_input(
        "Systolic BP", min_value=80, max_value=250, value=120
    )

    cholesterol = st.selectbox(
    "Cholesterol",
    [1, 2, 3],
    format_func=lambda x: {
        1: "Normal",
        2: "Above Normal",
        3: "Well Above Normal"
    }[x]
)

    smoke = st.selectbox(
        "Smoking",
        ["No", "Yes"]
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    weight = st.number_input(
        "Weight (kg)", min_value=30.0, max_value=200.0, value=65.0
    )

    ap_lo = st.number_input(
        "Diastolic BP", min_value=40, max_value=150, value=80
    )

    gluc = st.selectbox(
    "Glucose",
    [1, 2, 3],
    format_func=lambda x: {
        1: "Normal",
        2: "Above Normal",
        3: "Well Above Normal"
    }[x]
)

    alco = st.selectbox(
        "Alcohol",
        ["No", "Yes"]
    )

active = st.selectbox(
    "Physical Activity",
    ["No", "Yes"]
)

if st.button("Predict"):
    age_days = age_years * 365
    gender_value = 2 if gender == "Male" else 1
    smoke_value = 1 if smoke == "Yes" else 0
    alco_value = 1 if alco == "Yes" else 0
    active_value = 1 if active == "Yes" else 0
    bmi = weight / ((height / 100) ** 2)

    input_data = pd.DataFrame([[
        0, age_days, gender_value, height, weight,
        ap_hi, ap_lo, cholesterol, gluc,
        smoke_value, alco_value, active_value
    ]], columns=[
        "id", "age", "gender", "height", "weight",
        "ap_hi", "ap_lo", "cholesterol", "gluc",
        "smoke", "alco", "active"
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    if bmi < 18.5:
        bmi_status = "🔵 Underweight"
    elif bmi < 25:
        bmi_status = "🟢 Normal"
    elif bmi < 30:
        bmi_status = "🟠 Overweight"
    else:
        bmi_status = "🔴 Obese"

    st.info(f"""
### Patient BMI

**BMI:** {bmi:.2f}

**Category:** {bmi_status}
""")

    st.markdown("---")
    st.subheader("🧾 Patient Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Age:** {age_years} years")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Height:** {height} cm")
        st.write(f"**Weight:** {weight} kg")
        st.write(f"**BMI:** {bmi:.2f}")

    with col2:
        st.write(f"**Blood Pressure:** {ap_hi}/{ap_lo}")
        st.write(f"**Cholesterol:** {['Normal','Above Normal','Well Above Normal'][cholesterol-1]}")
        st.write(f"**Glucose:** {['Normal','Above Normal','Well Above Normal'][gluc-1]}")
        st.write(f"**Smoking:** {smoke}")
        st.write(f"**Alcohol:** {alco}")
        st.write(f"**Physical Activity:** {active}")

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if probability >= 70:
        st.error("🔴 High Cardiovascular Risk")
    elif probability >= 40:
        st.warning("🟡 Moderate Cardiovascular Risk")
    else:
        st.success("🟢 Low Cardiovascular Risk")

    st.metric("Risk Probability", f"{probability:.2f}%")
    st.progress(int(probability))

    st.markdown("---")
    st.subheader("💡 Health Recommendations")

    if probability >= 70:
        st.error("""
- Consult a healthcare professional.
- Monitor blood pressure regularly.
- Reduce salt and processed food intake.
- Exercise regularly.
- Avoid smoking and excessive alcohol.
""")
    elif probability >= 40:
        st.warning("""
- Maintain a healthy diet.
- Exercise at least 150 minutes per week.
- Get regular health checkups.
""")
    else:
        st.success("""
- Continue your healthy lifestyle.
- Maintain regular physical activity.
- Schedule routine health checkups.
""")

    st.warning("This app is for educational purposes only and is not medical advice.")