import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load(r"D:\GIT posting projects\Project-PCOS-Prediction\model\pcos_xgboost_model.joblib")
scaler = joblib.load(r"D:\GIT posting projects\Project-PCOS-Prediction\model\scaler.joblib")

# Page settings
st.set_page_config(page_title="HerHealth 🧬", layout="wide")

# ---------------- Sidebar ---------------- #

st.sidebar.header("ℹ About")
st.sidebar.markdown("""
This tool uses machine learning to screen for PCOS based on:
- *Ovarian morphology* (Follicle counts)  
- *Clinical symptoms* (Skin/hair changes)  
- *Metabolic factors* (BMI, diet, weight)  
- *Hormonal patterns* (LH/FSH ratio)  

*Model:* XGBoost (Accuracy: 92%)
""")

st.sidebar.title("ℹ How to Use")
st.sidebar.markdown("""
1. Enter patient details in the form.  
2. Fill both *symptoms* and *medical test values*.  
3. Click *Predict PCOS*.  
4. View results with *risk level, probability, and recommendations*.
""")

st.sidebar.title("✅ Ideal Values")
st.sidebar.markdown("""
- *BMI*: 18.5 – 24.9  
- *FSH/LH ratio*: LH should not be much higher than FSH  
- *Follicles*: < 12 per ovary  
- *Symptoms*: Absent (0 = No, 1 = Yes)  
""")


# ---------------- Title ---------------- #
st.markdown(
    "<h1 style='text-align: center; font-size: 40px; font-weight: bold;'>HerHealth 🧬</h1>",
    unsafe_allow_html=True
)
st.write("Enter patient details to check the likelihood of PCOS using the trained ML model.")

# ---------------- Input Section ---------------- #
with st.form("pcos_form"):
    st.subheader("📌 Patient Information")
    
    # Create three columns for input layout
    col1, col2, col3 = st.columns(3)
    
    # Column 1 - Basic Information
    with col1:
        st.markdown("### 📋 Basic Information")
        age = st.number_input("Age", min_value=15, max_value=50, value=25)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
        weight_gain = st.radio("Recent Weight Gain", ["No", "Yes"], horizontal=True)
        fast_food = st.radio("Frequent Fast Food Intake", ["No", "Yes"], horizontal=True)
    
    # Column 2 - Symptoms
    with col2:
        st.markdown("### 🧍 Symptoms")
        skin_darkening = st.radio("Skin Darkening", ["No", "Yes"], horizontal=True)
        hair_growth = st.radio("Excessive Hair Growth", ["No", "Yes"], horizontal=True)
        hair_loss = st.radio("Hair Loss", ["No", "Yes"], horizontal=True)
        pimples = st.radio("Pimples", ["No", "Yes"], horizontal=True)
    
    # Column 3 - Medical Values
    with col3:
        st.markdown("### 🩺 Medical Values")
        follicle_L = st.slider("Follicle Count (Left Ovary)", min_value=0, max_value=30, value=5)
        follicle_R = st.slider("Follicle Count (Right Ovary)", min_value=0, max_value=30, value=5)
        LH = st.number_input("LH (mIU/mL)", min_value=0.0, value=5.0, step=0.1)
        FSH = st.number_input("FSH (mIU/mL)", min_value=0.0, value=5.0, step=0.1)
    
    # Submit button
    submitted = st.form_submit_button("🔍 Predict PCOS", use_container_width=True)

# ---------------- Prediction ---------------- #
if submitted:
    # Convert Yes/No to binary 0/1
    weight_gain = 1 if weight_gain == "Yes" else 0
    fast_food = 1 if fast_food == "Yes" else 0
    skin_darkening = 1 if skin_darkening == "Yes" else 0
    hair_growth = 1 if hair_growth == "Yes" else 0
    hair_loss = 1 if hair_loss == "Yes" else 0
    pimples = 1 if pimples == "Yes" else 0
    
    # Feature engineering
    Total_Follicles = follicle_L + follicle_R
    Skin_Hair = skin_darkening + hair_growth + hair_loss + pimples
    Metabolic_Score = int(bmi >= 25) + int(fast_food == 1) + int(weight_gain == 1)
    High_LH = int((LH > FSH) and (hair_growth == 1))

    input_data = pd.DataFrame([[
        High_LH, Skin_Hair, Metabolic_Score, Total_Follicles
    ]], columns=['High_LH', 'Skin_Hair', 'Metabolic_Score', 'Total_Follicles'])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Store feature values for display
    features = {
        'Total_Follicles': Total_Follicles,
        'Skin_Hair': Skin_Hair,
        'Metabolic_Score': Metabolic_Score,
        'High_LH': High_LH
    }

    # Results layout
    st.subheader("📊 Prediction Results")
    
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.subheader("Risk Assessment")
        if prediction == 1:
            st.error(f"*High Risk of PCOS*")
            st.metric("Probability", f"{probability:.1%}")
        else:
            st.success(f"*Low Risk of PCOS*")
            st.metric("Probability", f"{probability:.1%}")
    
    with result_col2:
        st.subheader("Feature Scores")
        st.write(f"**Total Follicles:** {features['Total_Follicles']}")
        st.write(f"**Skin/Hair Symptoms:** {features['Skin_Hair']}/4")
        st.write(f"**Metabolic Score:** {features['Metabolic_Score']}/3")
        st.write(f"**High LH with Symptoms:** {'Yes' if features['High_LH'] == 1 else 'No'}")
    
    st.subheader("📋 Interpretation")
    if prediction == 1:
        st.warning("""
        **Recommendations:**  
        - Consult a gynecologist or endocrinologist  
        - Get ultrasound and hormonal tests confirmed  
        - Consider lifestyle changes (diet, exercise)  
        - Monitor menstrual cycle regularity  
        """)
    else:
        st.info("""
        **Recommendations:**  
        - Maintain healthy lifestyle  
        - Continue regular check-ups  
        - Monitor any new symptoms  
        """)