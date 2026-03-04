import streamlit as st
import numpy as np
import pandas as pd
import predict  # This now contains our "Manual Build" logic

st.set_page_config(page_title="Health Predictor", layout="wide")
st.title("🛡️ HEALTH PREDICTOR SYSTEM")

# --- REMOVED THE OLD LOAD_MODEL_ONCE ---
# We now use the 'my_model' that is already built inside predict.py

st.sidebar.header("📋 Patient Input")

# 1. Inputs (Keeping your exact logic)
age = st.sidebar.number_input("Age", 18, 100, 25)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 22.5)
s_bp = st.sidebar.number_input("Systolic Blood Pressure", 80 , 200, 120)
d_bp = st.sidebar.number_input("Diastolic Blood Pressure", 40, 130, 80)
stress = st.sidebar.slider("Stress Levels", 0,10,5)

gender = st.sidebar.radio("Gender", ["Male", "Female", "Other"])
smoke = st.sidebar.radio("Smoking Habits", ["Never", "Former", "Current"])
drink = st.sidebar.radio("Drinking Habits", ['Never', 'Occasional', 'Regular', 'Heavy'])
exercise = st.sidebar.radio("Exercise", ['Sedentary', 'Light', 'Moderate', 'Intense'])

db_hist = int(st.sidebar.toggle("Family History of Diabetes"))
hd_hist = int(st.sidebar.toggle("Family History of Heart Disease"))
ob_hist = int(st.sidebar.toggle("Family History of Obesity"))

# 2. Create the DataFrame (Matching the keys in predict.py)
raw_data = {
    'age': age,
    'gender': gender,
    'bmi': bmi,
    'systolic_bp': s_bp,
    'diastolic_bp': d_bp,
    'smoking': smoke,
    'drinking': drink,
    'exercise': exercise,
    'stress_level': stress,
    'family_history_diabetes': db_hist,
    'family_history_heart': hd_hist,
    'family_history_obesity': ob_hist
}
df = pd.DataFrame([raw_data])

# 3. Execution Logic
if st.button("Analyze Health Risks", type="primary"):
    with st.spinner("Analyzing data..."):
        # We call the new function from predict.py that handles 
        # both transformation AND the manual model prediction
        results = predict.make_predictions(df)
        
        st.divider()
        st.subheader("📊 Diagnostic Report")
        
        col1, col2, col3 = st.columns(3)
        
        # Displaying the results returned from predict.py
        col1.metric(label="Diabetes Risk", value=results["diabetes"])
        col2.metric(label="Heart Disease Risk", value=results["heart"])
        col3.metric(label="Obesity Risk", value=results["obesity"])
        
        st.info("Note: This is an AI-generated screening tool and not a clinical diagnosis.")