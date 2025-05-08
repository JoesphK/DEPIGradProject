import streamlit as st
import numpy as np
import pandas as pd
from model_loader import load_model

# Load models
models = {
    "Random Forest": load_model("models/HC_RandomForestModel.joblib"),
    "XGBoost": load_model("models/HC_XgBoostModel.joblib")
}

# Set background and styles
st.markdown(
    """
    <style>
    .stApp {
        background-color: cyan;
        color: black;
    }

    .stTextInput, .stNumberInput input {
        font-size: 26px;
    }

    .stTextInput, .stNumberInput p, .stSelectbox p {
        font-size: 32px;
    }

    .stElementContainer p {
        font-size: 24px;
    }

    .stButton {
        font-size: 32px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Diabetes Prediction")

# Model selection
model_choice = st.selectbox("Select Prediction Model", options=["Random Forest", "XGBoost"])
model = models[model_choice]

# Feature configurations: (label, type, default)
feature_config = [
    ("glucose", "numeric", 150, True),
    ("hypertensive", "boolean", False, False),
    ("diastolic_bp", "numeric", 80, False),
    ("systolic_bp", "numeric", 120, False),
    ("weight", "numeric", 80, True),
    ("bmi", "numeric", 20, True),
    ("age", "numeric", 30, False),
]


# Categorical mappings
select_mappings = {
    "gender": {"Male": 1, "Female": 0},
    "f_hypertension": {"No": 0, "Yes": 1},
    "family_diabetes": {"No": 0, "Yes": 1}
}

# Collect inputs, 3 per row
features = []
cols_per_row = 3

for i in range(0, len(feature_config), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, (label, ftype, default, isLog) in enumerate(feature_config[i:i + cols_per_row]):
        with cols[j]:
            if ftype == "numeric":
                val = st.number_input(label, value=default)
            elif ftype == "boolean":
                choice = st.selectbox(label, options=[True, False], index=int(not default))
                val = 1.0 if choice else 0.0
            elif ftype == "select":
                if label == "family_hypertension":
                    label = "f_hypertension"
                choice = st.selectbox(label, options=default)
                val = select_mappings[label][choice]

            if isLog:
                val = np.log1p(val)
            features.append(val)

# Predict button
if st.button("Predict"):
    feature_names = ["glucose_log", "hypertensive", "diastolic_bp", "systolic_bp", "weight_log", "bmi_log", "age"] 

    input_df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(input_df)
    print(prediction)
    if prediction[0] == True:
        st.error("You are diagnosed with diabetes.", icon="🚨")
    else:
        st.success("You are safe and not diagnosed with diabetes.", icon="✅")
