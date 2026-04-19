from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load saved files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = []

        # Map incoming data to the feature list used during training
        for feature in features:
            val = data.get(feature, 0)
            input_data.append(float(val))

        input_array = np.array(input_data).reshape(1, -1)
        
        # Transformation Pipeline
        scaled = scaler.transform(input_array)
        pca_data = pca.transform(scaled)

        # Prediction
        result = int(model.predict(pca_data)[0])
        prob = float(model.predict_proba(pca_data)[0][1])

        return jsonify({
            "risk": "High" if result == 1 else "Low",
            "probability": round(prob, 2),
            "components": int(pca.n_components_)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
