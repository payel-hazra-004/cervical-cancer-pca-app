#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# Input fields
input_data = []

for feature in features:
    val = st.number_input(f"{feature}", value=0.0)
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    scaled = scaler.transform(input_array)
    pca_data = pca.transform(scaled)

    result = model.predict(pca_data)
    prob = model.predict_proba(pca_data)[0][1]

    if result[0] == 1:
        st.error(f"⚠️ High Risk (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk (Probability: {prob:.2f})")

    st.write(f"Using {pca.n_components_} principal components")

