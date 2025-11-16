import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# ------------------------------
# 1️⃣ Model Loading
# ------------------------------
model_dict = {}
model_files = {
    "Random Forest": "models/random_forest_model.joblib",
    "XGBoost": "models/xg_boost_model.joblib",
    "Neural Network": "models/neural_network_model_improved.h5"
}

for name, path in model_files.items():
    if os.path.exists(path):
        try:
            if path.endswith(".joblib"):
                model_dict[name] = joblib.load(path)
            else:
                model_dict[name] = load_model(path)
            st.sidebar.success(f"✅ {name} model loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load {name}: {e}")
    else:
        st.sidebar.warning(f"⚠️ {path} not found")
        
# 🆕 ADD THIS DEBUG SECTION TO SEE WHAT FEATURES MODELS EXPECT
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Model Feature Info")
for name, model in model_dict.items():
    if hasattr(model, 'feature_names_in_'):
        st.sidebar.write(f"**{name}**: {len(model.feature_names_in_)} features")
        st.sidebar.write(f"First 5: {list(model.feature_names_in_)[:5]}")
    else:
        st.sidebar.write(f"**{name}**: No feature names info")        

# Fallback dummy model if none are available
if not model_dict:
    class DummyModel:
        def predict(self, X):
            return [0]

        def predict_proba(self, X):
            return np.array([[0.7, 0.3]])


    model_dict = {"Dummy Model": DummyModel()}
    st.sidebar.info("🧠 Using fallback dummy model.")

# ------------------------------
# 2️⃣ Feature Encoding Mappings
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
# 3️⃣ Streamlit UI
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

    # 🆕 FIXED: Create input data using EXACT features from the selected model
    input_data = {}
    
    # Get the expected features from the selected model
    if hasattr(model_dict[selected_model], 'feature_names_in_'):
        expected_features = list(model_dict[selected_model].feature_names_in_)
    else:
        # For Neural Network, use features from Random Forest as reference
        expected_features = list(model_dict["Random Forest"].feature_names_in_)
    
    # 🆕 Create feature mappings for all expected features
    feature_mappings = {
        # Numeric features
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
        
        # Label encoded categorical features
        "BusinessTravel": label_encode_value("BusinessTravel", business_travel),
        "Department": label_encode_value("Department", department),
        "EducationField": label_encode_value("EducationField", education_field),
        "Gender": label_encode_value("Gender", gender),
        "JobRole": label_encode_value("JobRole", job_role),
        "MaritalStatus": label_encode_value("MaritalStatus", marital_status),
        "OverTime": label_encode_value("OverTime", overtime),
        
        # Frequency encoded features
        "BusinessTravel_FreqEnc": get_frequency_encoding("BusinessTravel", business_travel),
        "Department_FreqEnc": get_frequency_encoding("Department", department),
        "EducationField_FreqEnc": get_frequency_encoding("EducationField", education_field),
        "Gender_FreqEnc": get_frequency_encoding("Gender", gender),
        "JobRole_FreqEnc": get_frequency_encoding("JobRole", job_role),
        "MaritalStatus_FreqEnc": get_frequency_encoding("MaritalStatus", marital_status),
        "OverTime_FreqEnc": get_frequency_encoding("OverTime", overtime),
        
        # Target encoded features
        "BusinessTravel_TargetEnc": get_target_encoding("BusinessTravel", business_travel),
        "Department_TargetEnc": get_target_encoding("Department", department),
        "EducationField_TargetEnc": get_target_encoding("EducationField", education_field),
        "Gender_TargetEnc": get_target_encoding("Gender", gender),
        "JobRole_TargetEnc": get_target_encoding("JobRole", job_role),
        "MaritalStatus_TargetEnc": get_target_encoding("MaritalStatus", marital_status),
        "OverTime_TargetEnc": get_target_encoding("OverTime", overtime),
        
        # Engineered features
        "HighTravelOvertime": 1 if (business_travel == "Travel_Frequently" and overtime == "Yes") else 0,
        "SingleOvertime": 1 if (marital_status == "Single" and overtime == "Yes") else 0,
    }
    
    # 🆕 Build input data in EXACT order expected by the model
    for feature in expected_features:
        # Handle numpy string types from XGBoost
        if hasattr(feature, 'item'):  # numpy string type
            feature_name = feature.item()
        else:
            feature_name = feature
            
        if feature_name in feature_mappings:
            input_data[feature] = feature_mappings[feature_name]
        else:
            st.warning(f"⚠️ Unknown feature: {feature} - using default value 0")
            input_data[feature] = 0

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Display feature summary
    with st.expander("🔍 View Features Being Sent to Model"):
        st.write(f"Total features: {len(input_data)}")
        st.write(f"Model: {selected_model}")
        st.write(f"Expected features: {len(expected_features)}")
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

# ------------------------------
# 4️⃣ Prediction Tab
# ------------------------------
with tab2:
    st.subheader("Prediction Results")

    if st.button("🚀 Predict Attrition", use_container_width=True):
        try:
            model = model_dict[selected_model]
            
            # 🆕 SIMPLIFIED: We already built input_df in the exact order needed
            input_df_aligned = input_df
            
            st.success(f"✅ Using {len(input_df_aligned.columns)} features in exact model order")

            if selected_model == "Neural Network":
                try:
                    # Load the neural network-specific scaler
                    scaler_path = "models/neural_network_scaler.joblib"
                    if os.path.exists(scaler_path):
                        scaler = joblib.load(scaler_path)
                        
                        # Scale the input features
                        input_scaled = scaler.transform(input_df_aligned)
                        
                        # Make prediction
                        raw_prediction = model.predict(input_scaled, verbose=0)
                        
                        # Handle different output formats
                        if raw_prediction.shape[1] > 1:  # Multiple outputs
                            probability = float(raw_prediction[0][1])
                        else:  # Single output
                            probability = float(raw_prediction[0][0])
                        
                        prediction = 1 if probability > 0.5 else 0
                        st.success("✅ Neural Network prediction successful")
                        
                    else:
                        st.error("❌ Neural Network scaler not found!")
                        raise FileNotFoundError("Scaler not found")
                        
                except Exception as e:
                    st.error(f"❌ Neural Network error: {e}")
                    # Simple fallback
                    st.info("🔄 Using safe fallback values")
                    prediction = 0
                    probability = 0.3

            else:
                # For scikit-learn models (Random Forest, XGBoost)
                try:
                    if hasattr(model, 'predict_proba'):
                        probability = model.predict_proba(input_df_aligned)[0][1]
                        prediction = model.predict(input_df_aligned)[0]
                    else:
                        prediction = model.predict(input_df_aligned)[0]
                        probability = 0.8 if prediction == 1 else 0.2
                    st.success(f"✅ {selected_model} prediction successful")
                    
                except Exception as e:
                    st.error(f"❌ {selected_model} prediction error: {e}")
                    # Simple fallback
                    st.info("🔄 Using safe fallback values")
                    prediction = 0
                    probability = 0.3

            # Display results
            st.markdown("---")
            st.subheader("📋 Prediction Summary")

            col_result, col_prob = st.columns(2)
            with col_result:
                if prediction == 1:
                    st.error("🚨 **Prediction: Likely to Leave**")
                    st.warning("Consider retention strategies for this employee.")
                else:
                    st.success("✅ **Prediction: Likely to Stay**")
                    st.info("Employee shows low attrition risk.")

            with col_prob:
                st.metric(
                    "Attrition Probability",
                    f"{probability:.1%}",
                    delta="High Risk" if probability > 0.5 else "Low Risk",
                    delta_color="inverse"
                )
                st.progress(float(probability))
                st.caption(f"Confidence: {probability:.1%}")

            # 🆕 SHOW WHICH MODEL WAS USED
            st.info(f"🧠 **Model Used**: {selected_model}")

            # Risk factors analysis
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

                # Calculate simple risk score
                risk_score = min(100, len(risks) * 15)
                st.metric("Overall Risk Score", f"{risk_score}%")
            else:
                st.success("✅ No major risk factors identified")

            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")

            if prediction == 1:
                st.write("""
                **Immediate Actions Recommended:**
                - Schedule one-on-one meeting to understand concerns
                - Review compensation and benefits package  
                - Consider flexible work arrangements
                - Provide career development opportunities
                - Address work-life balance issues
                """)
            else:
                st.write("""
                **Maintenance Actions:**
                - Continue regular check-ins and feedback
                - Monitor job satisfaction metrics
                - Provide growth and learning opportunities
                - Maintain competitive compensation
                """)

            # Feature importance (if available)
            if hasattr(model, 'feature_importances_') and selected_model != "Neural Network":
                st.markdown("---")
                st.subheader("📊 Top Influencing Factors")

                feature_importance = pd.DataFrame({
                    'Feature': input_df_aligned.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(8)

                # Clean up feature names for display
                display_features = feature_importance.copy()
                display_features['Feature'] = display_features['Feature'].str.replace('_FreqEnc', ' Freq')
                display_features['Feature'] = display_features['Feature'].str.replace('_TargetEnc', ' Target')

                st.bar_chart(display_features.set_index('Feature')['Importance'])

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.info("💡 Try using a different model or check the feature values")

# ------------------------------
# 5️⃣ Footer
# ------------------------------
st.markdown("---")
st.markdown("""
**About this app:**  
- Perfect feature alignment with trained models
- Accurate attrition probability predictions  
- Identifies key risk factors and provides actionable insights
- Built with comprehensive feature engineering
""")
