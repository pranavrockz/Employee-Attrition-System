import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# ------------------------------
# Config: model & scaler paths
# ------------------------------
MODEL_DIR = "models"
model_files = {
    "Random Forest": os.path.join(MODEL_DIR, "random_forest_model.joblib"),
    "XGBoost": os.path.join(MODEL_DIR, "xg_boost_model.joblib"),
    "Neural Network": os.path.join(MODEL_DIR, "neural_network_model_improved.h5")
}
nn_scaler_path = os.path.join(MODEL_DIR, "neural_network_scaler.joblib")

# ------------------------------
# EXPECTED FEATURES (exact order)
# Adjust this list to match the order used during training
# ------------------------------
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

# ------------------------------
# 1️⃣ Model Loading
# ------------------------------
model_dict = {}
for name, path in model_files.items():
    if os.path.exists(path):
        try:
            if path.endswith(".joblib"):
                model_dict[name] = joblib.load(path)
            else:
                # load keras model without compiling (safer for inference)
                model_dict[name] = load_model(path, compile=False)
            st.sidebar.success(f"✅ Loaded: {name}")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load {name}: {e}")
    else:
        st.sidebar.warning(f"⚠️ Not found: {path}")

# Load neural network scaler if present
nn_scaler = None
if os.path.exists(nn_scaler_path):
    try:
        nn_scaler = joblib.load(nn_scaler_path)
        st.sidebar.success("✅ Loaded neural network scaler")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load NN scaler: {e}")
else:
    st.sidebar.info("ℹ️ Neural network scaler not found")

# Fallback dummy model if none are available
if not model_dict:
    class DummyModel:
        def predict(self, X):
            return np.array([0])

        def predict_proba(self, X):
            return np.array([[0.7, 0.3]])

    model_dict = {"Dummy Model": DummyModel()}
    st.sidebar.info("🧠 Using fallback dummy model.")

# ------------------------------
# 2️⃣ Encoding helpers (same as training)
# ------------------------------
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


def label_encode_value(feature_name, value):
    if feature_name in categorical_mappings:
        try:
            return categorical_mappings[feature_name].index(value)
        except ValueError:
            return 0
    return value


def get_frequency_encoding(feature_name, value):
    return frequency_encodings.get(feature_name, {}).get(value, 0.1)


def get_target_encoding(feature_name, value):
    return target_encodings.get(feature_name, {}).get(value, 0.15)


# ------------------------------
# 3️⃣ Streamlit UI - Input Form
# ------------------------------
st.title("🏢 Employee Attrition Prediction")
st.markdown("Predict whether an employee is likely to leave based on input features.")

tab1, tab2 = st.tabs(["📊 Input Features", "🔮 Prediction Results"])

with tab1:
    st.subheader("Enter Employee Information")
    col1, col2 = st.columns(2)

    with col1:
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, 500)
        overtime = st.selectbox("Works Overtime", ["No", "Yes"])
        department = st.selectbox("Department", ["HR", "R&D", "Sales"])
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        education_field = st.selectbox("Education Field", ["Life Sciences", "Technical", "Medical", "Marketing"])

    with col2:
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        bonus = st.number_input("Bonus ($)", 0, 50000, 500, 100)
        work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
        job_level = st.slider("Job Level", 1, 5, 2)
        stock_option_level = st.slider("Stock Option Level", 0, 3, 1)

    col3, col4 = st.columns(2)
    with col3:
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        distance_from_home = st.slider("Distance From Home (miles)", 1, 50, 10)
        job_involvement = st.slider("Job Involvement", 1, 4, 3)
        performance_rating = st.slider("Performance Rating", 1, 4, 3)

    with col4:
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
        num_companies_worked = st.slider("Number of Companies Worked", 0, 10, 2)
        total_working_years = st.slider("Total Working Years", 0, 40, 8)
        training_times_last_year = st.slider("Training Times Last Year", 0, 6, 3)

    col5, col6 = st.columns(2)
    with col5:
        gender = st.selectbox("Gender", ["Female", "Male"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    with col6:
        education = st.slider("Education Level", 1, 5, 3)
        hourly_rate = st.number_input("Hourly Rate ($)", 30, 200, 65)
        job_role = st.selectbox("Job Role",
                                ["HR", "Lab Technician", "Manager", "Research Scientist", "Sales Executive"])

    st.markdown("---")
    selected_model = st.selectbox("Choose Model", list(model_dict.keys()))

    # Build feature dict according to EXPECTED_FEATURES
    feature_map = {
        # numeric
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
        "EmployeeNumber": 1001,  # placeholder
        # label encoded
        "BusinessTravel": label_encode_value("BusinessTravel", business_travel),
        "Department": label_encode_value("Department", department),
        "EducationField": label_encode_value("EducationField", education_field),
        "Gender": label_encode_value("Gender", gender),
        "JobRole": label_encode_value("JobRole", job_role),
        "MaritalStatus": label_encode_value("MaritalStatus", marital_status),
        "OverTime": label_encode_value("OverTime", overtime),
        # freq enc
        "BusinessTravel_FreqEnc": get_frequency_encoding("BusinessTravel", business_travel),
        "Department_FreqEnc": get_frequency_encoding("Department", department),
        "EducationField_FreqEnc": get_frequency_encoding("EducationField", education_field),
        "Gender_FreqEnc": get_frequency_encoding("Gender", gender),
        "JobRole_FreqEnc": get_frequency_encoding("JobRole", job_role),
        "MaritalStatus_FreqEnc": get_frequency_encoding("MaritalStatus", marital_status),
        "OverTime_FreqEnc": get_frequency_encoding("OverTime", overtime),
        # target enc
        "BusinessTravel_TargetEnc": get_target_encoding("BusinessTravel", business_travel),
        "Department_TargetEnc": get_target_encoding("Department", department),
        "EducationField_TargetEnc": get_target_encoding("EducationField", education_field),
        "Gender_TargetEnc": get_target_encoding("Gender", gender),
        "JobRole_TargetEnc": get_target_encoding("JobRole", job_role),
        "MaritalStatus_TargetEnc": get_target_encoding("MaritalStatus", marital_status),
        "OverTime_TargetEnc": get_target_encoding("OverTime", overtime),
        # engineered
        "HighTravelOvertime": 1 if (business_travel == "Travel_Frequently" and overtime == "Yes") else 0,
        "SingleOvertime": 1 if (marital_status == "Single" and overtime == "Yes") else 0,
    }

    # Ensure all expected features exist (fill missing with 0)
    input_data = {f: feature_map.get(f, 0) for f in EXPECTED_FEATURES}
    input_df = pd.DataFrame([input_data])[EXPECTED_FEATURES]

    with st.expander("🔍 View Features Being Sent to Model"):
        st.write(f"Total features: {len(input_df.columns)}")
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

# ------------------------------
# 4️⃣ Prediction Tab
# ------------------------------
with tab2:
    st.subheader("Prediction Results")

    if st.button("🚀 Predict Attrition", use_container_width=True):
        try:
            model = model_dict[selected_model]

            # Align input for models that expose feature_names_in_
            input_df_aligned = input_df.copy()
            if hasattr(model, "feature_names_in_"):
                # ensure the order and exact names match
                expected = list(model.feature_names_in_)
                # if any expected feature missing, fill with zeros
                for feat in expected:
                    if feat not in input_df_aligned.columns:
                        input_df_aligned[feat] = 0
                input_df_aligned = input_df_aligned[expected]
            else:
                # otherwise, use EXPECTED_FEATURES order (already ensured above)
                input_df_aligned = input_df_aligned[EXPECTED_FEATURES]

            st.success(f"✅ Prepared input with {input_df_aligned.shape[1]} features")

            # Predict
            if selected_model == "Neural Network":
                if nn_scaler is None:
                    st.error("❌ NN scaler not loaded. Cannot run neural network. Falling back.")
                    # fallback
                    if "Random Forest" in model_dict:
                        rf = model_dict["Random Forest"]
                        probability = rf.predict_proba(input_df_aligned)[0][1]
                        prediction = int(rf.predict(input_df_aligned)[0])
                    else:
                        probability = 0.3
                        prediction = 0
                else:
                    # Convert to numpy and scale
                    input_np = input_df_aligned.to_numpy()
                    input_scaled = nn_scaler.transform(input_np)
                    raw = model.predict(input_scaled, verbose=0)
                    # raw might be shape (1,1) or (1,) depending on model saving — handle both
                    if isinstance(raw, np.ndarray):
                        probability = float(np.asarray(raw).reshape(-1)[0])
                    else:
                        probability = float(raw[0])
                    prediction = 1 if probability > 0.5 else 0
                    st.success("✅ Neural Network prediction successful")

            else:
                # scikit-learn models
                if hasattr(model, "predict_proba"):
                    probability = float(model.predict_proba(input_df_aligned)[0][1])
                    prediction = int(model.predict(input_df_aligned)[0])
                else:
                    prediction = int(model.predict(input_df_aligned)[0])
                    probability = 0.8 if prediction == 1 else 0.2

            # Display results
            st.markdown("---")
            st.subheader("📋 Prediction Summary")
            col_result, col_prob = st.columns(2)
            with col_result:
                if prediction == 1:
                    st.error("🚨 **Prediction: Likely to Leave**")
                else:
                    st.success("✅ **Prediction: Likely to Stay**")

            with col_prob:
                st.metric(
                    "Attrition Probability",
                    f"{probability:.1%}",
                    delta="High Risk" if probability > 0.5 else "Low Risk",
                    delta_color="inverse"
                )
                st.progress(float(probability))
                st.caption(f"Confidence: {probability:.1%}")

            # Risk factors (same logic as before)
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
                risks.append("💰 Below Average Income")
            if environment_satisfaction <= 2:
                risks.append("🏢 Poor Environment Satisfaction")
            if business_travel == "Travel_Frequently":
                risks.append("✈️ Frequent Business Travel")
            if relationship_satisfaction <= 2:
                risks.append("👥 Poor Relationship Satisfaction")

            if risks:
                st.write("**Identified Risk Factors:**")
                for risk in risks:
                    st.write(f"• {risk}")
                risk_score = min(100, len(risks) * 15)
                st.metric("Overall Risk Score", f"{risk_score}%")
            else:
                st.success("✅ No major risk factors identified")

            # Feature importance for tree models
            if hasattr(model, 'feature_importances_') and selected_model != "Neural Network":
                st.markdown("---")
                st.subheader("📊 Top Influencing Factors")
                fi = pd.DataFrame({
                    'Feature': input_df_aligned.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(8)
                fi['Feature'] = fi['Feature'].str.replace('_FreqEnc', ' Freq').str.replace('_TargetEnc', ' Target')
                st.bar_chart(fi.set_index('Feature')['Importance'])

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("""
**About this app:**  
- Input features aligned to the training order  
- Neural Network uses a saved StandardScaler for exact scaling  
- Fallbacks in place if model or scaler files are missing
""")
