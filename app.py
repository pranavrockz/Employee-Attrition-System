import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# ----------------------------------------------------------
# Streamlit Page Settings
# ----------------------------------------------------------
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# ----------------------------------------------------------
# Paths for Models and Scaler
# ----------------------------------------------------------
MODEL_DIR = "models"

model_files = {
    "Random Forest": os.path.join(MODEL_DIR, "random_forest_model.joblib"),
    "XGBoost": os.path.join(MODEL_DIR, "xg_boost_model.joblib"),
    "Neural Network": os.path.join(MODEL_DIR, "neural_network_model_improved.h5")
}

nn_scaler_path = os.path.join(MODEL_DIR, "neural_network_scaler.joblib")

# ----------------------------------------------------------
# Expected Features (same as during training!)
# ----------------------------------------------------------
EXPECTED_FEATURES = [
    'BusinessTravel', 'Department', 'DistanceFromHome', 'Education',
    'EducationField', 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole',
    'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'Bonus',
    'NumCompaniesWorked', 'OverTime', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'BusinessTravel_FreqEnc',
    'Department_FreqEnc', 'EducationField_FreqEnc', 'Gender_FreqEnc',
    'JobRole_FreqEnc', 'MaritalStatus_FreqEnc', 'OverTime_FreqEnc',
    'BusinessTravel_TargetEnc', 'Department_TargetEnc',
    'EducationField_TargetEnc', 'Gender_TargetEnc', 'JobRole_TargetEnc',
    'MaritalStatus_TargetEnc', 'OverTime_TargetEnc', 'HighTravelOvertime',
    'SingleOvertime'
]

# ----------------------------------------------------------
# Load Models
# ----------------------------------------------------------
st.sidebar.header("Model Status")

model_dict = {}
for name, path in model_files.items():
    if os.path.exists(path):
        try:
            if path.endswith(".joblib"):
                model_dict[name] = joblib.load(path)
            else:
                model_dict[name] = load_model(path, compile=False)
            st.sidebar.success(f"Loaded: {name}")
        except Exception as e:
            st.sidebar.error(f"Error loading {name}: {e}")
    else:
        st.sidebar.warning(f"Not Found: {path}")

# Load NN scaler
nn_scaler = None
if os.path.exists(nn_scaler_path):
    try:
        nn_scaler = joblib.load(nn_scaler_path)
        st.sidebar.success("Loaded NN scaler")
    except Exception as e:
        st.sidebar.error(f"Scaler load error: {e}")
else:
    st.sidebar.warning("NN scaler not found")

# ----------------------------------------------------------
# Categorical Mappings (Label encoding, frequency encoding, target encoding)
# ----------------------------------------------------------
categorical_mappings = {
    'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
    'Department': ['HR', 'R&D', 'Sales'],
    'EducationField': ['Life Sciences', 'Marketing', 'Medical', 'Technical'],
    'Gender': ['Female', 'Male'],
    'JobRole': ['HR', 'Lab Technician', 'Manager', 'Research Scientist', 'Sales Executive'],
    'MaritalStatus': ['Divorced', 'Married', 'Single'],
    'OverTime': ['No', 'Yes']
}

frequency_encodings = {
    'BusinessTravel': {'Non-Travel': 0.1, 'Travel_Rarely': 0.6, 'Travel_Frequently': 0.3},
    'Department': {'HR': 0.2, 'R&D': 0.4, 'Sales': 0.4},
    'EducationField': {'Life Sciences': 0.3, 'Marketing': 0.2, 'Medical': 0.25, 'Technical': 0.25},
    'Gender': {'Female': 0.45, 'Male': 0.55},
    'JobRole': {'HR': 0.1, 'Lab Technician': 0.2, 'Manager': 0.15, 'Research Scientist': 0.3, 'Sales Executive': 0.25},
    'MaritalStatus': {'Single': 0.3, 'Married': 0.5, 'Divorced': 0.2},
    'OverTime': {'No': 0.7, 'Yes': 0.3}
}

target_encodings = {
    'BusinessTravel': {'Non-Travel': 0.08, 'Travel_Rarely': 0.12, 'Travel_Frequently': 0.28},
    'Department': {'HR': 0.25, 'R&D': 0.12, 'Sales': 0.18},
    'EducationField': {'Life Sciences': 0.14, 'Marketing': 0.16, 'Medical': 0.12, 'Technical': 0.15},
    'Gender': {'Female': 0.14, 'Male': 0.16},
    'JobRole': {'HR': 0.22, 'Lab Technician': 0.15, 'Manager': 0.08, 'Research Scientist': 0.12,
                'Sales Executive': 0.20},
    'MaritalStatus': {'Single': 0.22, 'Married': 0.12, 'Divorced': 0.18},
    'OverTime': {'No': 0.12, 'Yes': 0.25}
}

def label_encode_value(feature, value):
    return categorical_mappings.get(feature, []).index(value) \
        if value in categorical_mappings.get(feature, []) else 0

def get_frequency_encoding(feature, value):
    return frequency_encodings.get(feature, {}).get(value, 0.1)

def get_target_encoding(feature, value):
    return target_encodings.get(feature, {}).get(value, 0.15)

# ----------------------------------------------------------
# UI - Input Form
# ----------------------------------------------------------
st.title("🏢 Employee Attrition Prediction")

tab1, tab2 = st.tabs(["📊 Input Features", "🔮 Prediction Results"])

with tab1:
    st.subheader("Employee Information")
    col1, col2 = st.columns(2)

    with col1:
        monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        department = st.selectbox("Department", ["HR", "R&D", "Sales"])
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        education_field = st.selectbox("Education Field", ["Life Sciences", "Technical", "Medical", "Marketing"])

    with col2:
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        bonus = st.number_input("Bonus", 0, 50000, 500)
        work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
        job_level = st.slider("Job Level", 1, 5, 2)
        stock_option_level = st.slider("Stock Option Level", 0, 3, 1)

    col3, col4 = st.columns(2)

    with col3:
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        distance_from_home = st.slider("Distance From Home", 1, 50, 10)
        job_involvement = st.slider("Job Involvement", 1, 4, 3)
        performance_rating = st.slider("Performance Rating", 1, 4, 3)

    with col4:
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
        num_companies_worked = st.slider("Companies Worked", 0, 10, 2)
        total_working_years = st.slider("Total Working Years", 0, 40, 8)
        training_times_last_year = st.slider("Training Times Last Year", 0, 6, 3)

    col5, col6 = st.columns(2)

    with col5:
        gender = st.selectbox("Gender", ["Female", "Male"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    with col6:
        education = st.slider("Education Level", 1, 5, 3)
        hourly_rate = st.number_input("Hourly Rate", 30, 200, 65)
        job_role = st.selectbox("Job Role", 
            ["HR", "Lab Technician", "Manager", "Research Scientist", "Sales Executive"])

    selected_model = st.selectbox("Choose Model", list(model_dict.keys()))

    # ----------------------------------------------------------
    # Build Feature Dictionary
    # ----------------------------------------------------------
    feature_map = {
        "MonthlyIncome": monthly_income,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EnvironmentSatisfaction": environment_satisfaction,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "Bonus": bonus,
        "NumCompaniesWorked": num_companies_worked,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "EmployeeNumber": 1001,

        "BusinessTravel": label_encode_value("BusinessTravel", business_travel),
        "Department": label_encode_value("Department", department),
        "EducationField": label_encode_value("EducationField", education_field),
        "Gender": label_encode_value("Gender", gender),
        "JobRole": label_encode_value("JobRole", job_role),
        "MaritalStatus": label_encode_value("MaritalStatus", marital_status),
        "OverTime": label_encode_value("OverTime", overtime),

        "BusinessTravel_FreqEnc": get_frequency_encoding("BusinessTravel", business_travel),
        "Department_FreqEnc": get_frequency_encoding("Department", department),
        "EducationField_FreqEnc": get_frequency_encoding("EducationField", education_field),
        "Gender_FreqEnc": get_frequency_encoding("Gender", gender),
        "JobRole_FreqEnc": get_frequency_encoding("JobRole", job_role),
        "MaritalStatus_FreqEnc": get_frequency_encoding("MaritalStatus", marital_status),
        "OverTime_FreqEnc": get_frequency_encoding("OverTime", overtime),

        "BusinessTravel_TargetEnc": get_target_encoding("BusinessTravel", business_travel),
        "Department_TargetEnc": get_target_encoding("Department", department),
        "EducationField_TargetEnc": get_target_encoding("EducationField", education_field),
        "Gender_TargetEnc": get_target_encoding("Gender", gender),
        "JobRole_TargetEnc": get_target_encoding("JobRole", job_role),
        "MaritalStatus_TargetEnc": get_target_encoding("MaritalStatus", marital_status),
        "OverTime_TargetEnc": get_target_encoding("OverTime", overtime),

        "HighTravelOvertime":
            1 if (business_travel == "Travel_Frequently" and overtime == "Yes") else 0,

        "SingleOvertime":
            1 if (marital_status == "Single" and overtime == "Yes") else 0
    }

    # Build input DataFrame with correct order
    input_data = {f: feature_map.get(f, 0) for f in EXPECTED_FEATURES}
    input_df = pd.DataFrame([input_data])[EXPECTED_FEATURES]

# ----------------------------------------------------------
# Prediction Tab
# ----------------------------------------------------------
with tab2:
    st.subheader("Prediction Results")

    if st.button("🚀 Predict Attrition", use_container_width=True):

        try:
            model = model_dict[selected_model]

            # Force correct order
            input_aligned = input_df.copy()

            # Some models have feature_names_in_
            if hasattr(model, "feature_names_in_"):
                expected = list(model.feature_names_in_)
                for col in expected:
                    if col not in input_aligned:
                        input_aligned[col] = 0
                input_aligned = input_aligned[expected]

            # Predict
            if selected_model == "Neural Network":
                if nn_scaler is None:
                    st.error("Scaler missing for Neural Network!")
                    raise ValueError("NN Scaler Missing")

                data_np = input_aligned.to_numpy()
                data_scaled = nn_scaler.transform(data_np)

                raw = model.predict(data_scaled, verbose=0)
                probability = float(np.asarray(raw).reshape(-1)[0])
                probability = max(0.01, min(0.99, probability))  # avoid 100%
                prediction = 1 if probability > 0.5 else 0

            else:
                if hasattr(model, "predict_proba"):
                    probability = float(model.predict_proba(input_aligned)[0][1])
                    prediction = int(model.predict(input_aligned)[0])
                else:
                    prediction = int(model.predict(input_aligned)[0])
                    probability = 0.75 if prediction else 0.25

            st.markdown("### 📋 Prediction Summary")
            colL, colR = st.columns(2)

            with colL:
                if prediction == 1:
                    st.error("🚨 Likely to Leave")
                else:
                    st.success("✅ Likely to Stay")

            with colR:
                st.metric("Attrition Probability", f"{probability:.1%}",
                          "High Risk" if probability > 0.5 else "Low Risk")
                st.progress(probability)

            # --------------------------------------------
            # Risk Explanation
            # --------------------------------------------
            st.markdown("---")
            st.subheader("🔍 Key Risk Factors")
            risks = []

            if overtime == "Yes":
                risks.append("🚨 Works Overtime")
            if job_satisfaction <= 2:
                risks.append("😞 Low Job Satisfaction")
            if work_life_balance <= 2:
                risks.append("⚖️ Poor Work-Life Balance")
            if monthly_income < 4500:
                risks.append("💰 Low Monthly Income")
            if environment_satisfaction <= 2:
                risks.append("🏢 Poor Environment Satisfaction")
            if business_travel == "Travel_Frequently":
                risks.append("✈️ Frequent Business Travel")
            if relationship_satisfaction <= 2:
                risks.append("👥 Poor Relationship Satisfaction")

            if risks:
                st.write("**Identified Risks:**")
                for r in risks:
                    st.write("• " + r)
                st.metric("Overall Risk Score", f"{min(100, len(risks)*15)}%")
            else:
                st.success("No major risk factors identified")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# Footer
st.markdown("---")
st.caption("App uses consistent preprocessing, feature ordering, and NN scaling for accurate predictions.")
