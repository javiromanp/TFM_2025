from flask import Flask, request, jsonify, render_template, redirect, url_for
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.generic")

from dependencies.preprocessing import preprocess_text
from dependencies.model_utils import get_final_prediction_ensemble

app = Flask(__name__)

MODEL_PATHS = [
    "models/xlm_hp_3_1",
    "models/beto_hp_3_2",
    "models/roberta-large_hp_3_3"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/alert')
def alert():
    return render_template('alert.html')

@app.route("/device")
def get_device():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return jsonify({"device": device})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    if text:
        phrases = [phrase.strip() for phrase in text.split('\n') if phrase.strip()]
        if len(phrases) > 10:
            return jsonify({"error": "Se permiten hasta 10 frases."}), 400

        preprocessed_phrases = [preprocess_text(phrase) for phrase in phrases]
        final_classes = []
        avg_scores_list = []

        try:
            for phrase in preprocessed_phrases:
                final_class, avg_scores = get_final_prediction_ensemble(phrase)
                final_classes.append(int(final_class))
                avg_scores_list.append(avg_scores.tolist())

            response = {
                "message": "Textos recibidos y preprocesados",
                "texts": preprocessed_phrases,
                "classes": final_classes,
                "scores": avg_scores_list
            }
            return jsonify(response)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)

