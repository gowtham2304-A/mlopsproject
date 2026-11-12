
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "best_model.pkl"
)

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
imputer = saved_data["imputer"]
scaler = saved_data["scaler"]
features = saved_data["features"]
model_name = saved_data["model_name"]

print("Model loaded successfully!")
print("Selected model:", model_name)

@app.route("/")
def home():
    return render_template(
        "index.html",
        features=features
    )

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "message": "API is working"
    })

@app.route("/model-info")
def model_info():
    return jsonify({
        "model_name": model_name,
        "number_of_features": len(features),
        "features": features
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        customer = request.get_json()

        if not isinstance(customer, dict):
            return jsonify({
                "error": "Send customer data as JSON"
            }), 400
        missing_features = [
            feature for feature in features
            if feature not in customer
        ]

        if missing_features:
            return jsonify({
                "error": "Missing features",
                "missing_features": missing_features
            }), 400
        input_data = pd.DataFrame(
            [[customer[feature] for feature in features]],
            columns=features
        )

        input_data = input_data.apply(
            pd.to_numeric,
            errors="coerce"
        )
        if input_data.isnull().any().any():
            return jsonify({
                "error": "Please enter valid numbers for all fields"
            }), 400
        input_data = imputer.transform(input_data)
        input_data = scaler.transform(input_data)
        prediction = int(model.predict(input_data)[0])
        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        result = (
            "Default Predicted"
            if prediction == 1
            else "No Default Predicted"
        )
        return jsonify({
            "prediction": prediction,
            "result": result,
            "default_probability": round(
                probability * 100, 2
            ),
            "model": model_name
        })

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    try:
        request_data = request.get_json()

        if not isinstance(request_data, dict):
            return jsonify({
                "error": "Send a JSON object"
            }), 400
        customers = request_data.get("customers")
        if not isinstance(customers, list) or not customers:
            return jsonify({
                "error": "Provide a non-empty customers list"
            }), 400
        if not all(isinstance(c, dict) for c in customers):
            return jsonify({
                "error": "Every customer must be an object"
            }), 400
        input_data = pd.DataFrame(customers)
        missing_features = [
            feature for feature in features
            if feature not in input_data.columns
        ]

        if missing_features:
            return jsonify({
                "error": "Missing features",
                "missing_features": missing_features
            }), 400
        input_data = input_data[features]
        input_data = input_data.apply(
            pd.to_numeric,
            errors="coerce"
        )
        if input_data.isnull().any().any():
            return jsonify({
                "error": "All feature values must be numbers"
            }), 400
        input_data = imputer.transform(input_data)
        input_data = scaler.transform(input_data)
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]
        results = []
        for prediction, probability in zip(
            predictions, probabilities
        ):
            results.append({
                "prediction": int(prediction),
                "result": (
                    "Default Predicted"
                    if prediction == 1
                    else "No Default Predicted"
                ),
                "default_probability": round(
                    float(probability) * 100, 2
                )
            })
        return jsonify({
            "total_customers": len(results),
            "results": results
        })

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
