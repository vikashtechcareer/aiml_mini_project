# app/app.py
from flask import Flask, request, jsonify
import joblib, os, random
import pandas as pd

app = Flask(__name__)

# Auto-load all models from directory
def load_models(dataset_name):
    model_dir = f"../models/{dataset_name}_models"
    models = {}
    for file in os.listdir(model_dir):
        if file.endswith('.joblib'):
            name = file.replace('.joblib', '')
            models[name] = joblib.load(os.path.join(model_dir, file))
    return models

# Load all models into memory
model_store = {
    'iris': load_models('iris'),
    'titanic': load_models('titanic')
}

@app.route('/predict/<dataset>/manual/<model_name>', methods=['POST'])
def predict_manual(dataset, model_name):
    data = request.json
    if dataset not in model_store or model_name not in model_store[dataset]:
        return jsonify({"error": "Invalid dataset or model name"}), 400

    try:
        df = pd.DataFrame([data])
        prediction = model_store[dataset][model_name].predict(df)[0]
        return jsonify({
            "model_used": model_name,
            "prediction": int(prediction)
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict/<dataset>/ab', methods=['POST'])
def predict_ab(dataset):
    data = request.json
    model_a = request.args.get('model_a')
    model_b = request.args.get('model_b')

    if not model_a or not model_b:
        return jsonify({"error": "Please provide 'model_a' and 'model_b' as query params"}), 400
    if model_a not in model_store[dataset] or model_b not in model_store[dataset]:
        return jsonify({"error": "Invalid model names"}), 400

    variant = random.choice(['a', 'b'])
    selected_model = model_store[dataset][model_a if variant == 'a' else model_b]

    try:
        df = pd.DataFrame([data])
        prediction = selected_model.predict(df)[0]
        return jsonify({
            "variant": variant,
            "model_used": model_a if variant == 'a' else model_b,
            "prediction": int(prediction)
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict/<dataset>/canary', methods=['POST'])
def predict_canary(dataset):
    data = request.json
    model_a = request.args.get('model_a')
    model_b = request.args.get('model_b')
    weight_a = float(request.args.get('weight_a', 0.9))
    weight_b = float(request.args.get('weight_b', 0.1))

    if not model_a or not model_b:
        return jsonify({"error": "Provide 'model_a' and 'model_b' as query params"}), 400
    if model_a not in model_store[dataset] or model_b not in model_store[dataset]:
        return jsonify({"error": "Invalid model names"}), 400

    variant = random.choices(['a', 'b'], weights=[weight_a, weight_b])[0]
    selected_model = model_store[dataset][model_a if variant == 'a' else model_b]

    try:
        df = pd.DataFrame([data])
        prediction = selected_model.predict(df)[0]
        return jsonify({
            "variant": variant,
            "model_used": model_a if variant == 'a' else model_b,
            "weights": {"a": weight_a, "b": weight_b},
            "prediction": int(prediction)
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
