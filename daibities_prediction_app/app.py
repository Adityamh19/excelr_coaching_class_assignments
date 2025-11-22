import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ----------------------------------------
# Load Model and Scaler
# ----------------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ----------------------------------------
# Streamlit UI
# ----------------------------------------
st.title("🩺 Diabetes Prediction App")
st.write("Enter the patient details below. The model will make a prediction and also explain how your inputs impact the result.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]])

columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# ----------------------------------------
# Explanation logic
# ----------------------------------------
def generate_explanation(row):
    explanations = []

    if row["Glucose"] > 140:
        explanations.append("High glucose strongly increases diabetes risk.")

    if row["BMI"] > 30:
        explanations.append("Higher BMI indicates obesity, which impacts insulin resistance.")

    if row["Insulin"] > 200:
        explanations.append("High insulin levels suggest impaired glucose regulation.")

    if row["Age"] > 45:
        explanations.append("Risk increases with age due to metabolic slowdown.")

    if row["BloodPressure"] > 90:
        explanations.append("High blood pressure is linked to insulin resistance.")

    if row["Pregnancies"] > 5:
        explanations.append("More pregnancies increase long-term hormonal stress.")

    if row["DiabetesPedigreeFunction"] > 1.0:
        explanations.append("Higher DPF shows strong genetic influence.")

    if len(explanations) == 0:
        return "Your values are mostly within a normal range, reducing diabetes risk."

    return " | ".join(explanations)


# ----------------------------------------
# Prediction Button
# ----------------------------------------
if st.button("Predict"):
    # Scale and predict
    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    # Create output table
    df = pd.DataFrame(input_data, columns=columns)
    df["Prediction"] = "Diabetic" if prediction == 1 else "Not Diabetic"
    df["Probability"] = round(prob, 3)

    # Display table
    st.subheader("📊 Input Summary + Model Output")
    st.dataframe(df)

    # Explanation
    st.subheader("🧠 Why These Inputs Matter")
    explanation = generate_explanation(df.iloc[0])
    st.write(explanation)

    # Final result styling
    if prediction == 1:
        st.error(f"🔴 **Diabetes Detected** (Probability: {prob:.2f})")
    else:
        st.success(f"🟢 **No Diabetes** (Probability: {prob:.2f})")
