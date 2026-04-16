#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("kag_risk_factors_cervical_cancer.csv")

# -------------------------------
# Preprocessing
# -------------------------------
df.replace("?", np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.median(), inplace=True)

# -------------------------------
# Features & Target
# -------------------------------
X = df.drop("Biopsy", axis=1)
y = df["Biopsy"]

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# PCA (95% variance)
# -------------------------------
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print("Original features:", X.shape[1])
print("Reduced components:", X_pca.shape[1])

# -------------------------------
# Train model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Save everything
# -------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(pca, open("pca.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("features.pkl", "wb"))

print("✅ All files saved!")

