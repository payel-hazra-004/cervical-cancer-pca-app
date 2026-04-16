import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Load saved files
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Cervical Cancer Predictor")

st.title("🧬 Cervical Cancer Prediction (PCA Model)")
st.write("Enter patient details:")

# -------------------------------
# User Inputs (Reduced & Manual)
# -------------------------------
user_inputs = {}

user_inputs["Age"] = st.text_input("Age")
user_inputs["Number of sexual partners"] = st.text_input("Sexual partners")
user_inputs["First sexual intercourse"] = st.text_input("Age at first intercourse")
user_inputs["Num of pregnancies"] = st.text_input("Pregnancies")

user_inputs["Smokes"] = st.selectbox("Do you smoke?", ["Select", "No", "Yes"])
user_inputs["Hormonal Contraceptives"] = st.selectbox("Use hormonal contraceptives?", ["Select", "No", "Yes"])
user_inputs["IUD"] = st.selectbox("Use IUD?", ["Select", "No", "Yes"])
user_inputs["STDs"] = st.selectbox("Any STD history?", ["Select", "No", "Yes"])
user_inputs["HPV"] = st.selectbox("HPV infection?", ["Select", "No", "Yes"])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    try:
        # Convert numeric inputs
        user_inputs["Age"] = float(user_inputs["Age"])
        user_inputs["Number of sexual partners"] = float(user_inputs["Number of sexual partners"])
        user_inputs["First sexual intercourse"] = float(user_inputs["First sexual intercourse"])
        user_inputs["Num of pregnancies"] = float(user_inputs["Num of pregnancies"])

        # Convert Yes/No inputs
        for key in ["Smokes", "Hormonal Contraceptives", "IUD", "STDs", "HPV"]:
            if user_inputs[key] == "Select":
                st.warning(f"Please select {key}")
                st.stop()
            user_inputs[key] = 1 if user_inputs[key] == "Yes" else 0

    except:
        st.error("Please enter valid numeric values")
        st.stop()

    # -------------------------------
    # Create input data (FIXED)
    # -------------------------------
    input_data = []

    for feature in features:
        if feature in user_inputs:
            input_data.append(user_inputs[feature])
        else:
            input_data.append(0)

    input_array = np.array(input_data).reshape(1, -1)

    # -------------------------------
    # Prediction
    # -------------------------------
    scaled = scaler.transform(input_array)
    pca_data = pca.transform(scaled)

    result = model.predict(pca_data)
    prob = model.predict_proba(pca_data)[0][1]

    if result[0] == 1:
        st.error(f"⚠️ High Risk (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk (Probability: {prob:.2f})")

    st.write(f"Using {pca.n_components_} principal components")
