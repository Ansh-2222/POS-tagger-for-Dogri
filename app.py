from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import os

# Load model files
clf = joblib.load("model_files/model.pkl")
le = joblib.load("model_files/label_encoder.pkl")
vec = joblib.load("model_files/vectorizer.pkl")

app = Flask(__name__)
CORS(app)  # Allow remote calls

# Serve frontend
@app.route("/")
def home():
    return render_template("index.html")

# API for POS tagging
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sentence = data.get("sentence", "")
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    tokens = sentence.split()
    tagged = []
    for idx, word in enumerate(tokens):
        feats = {
            "word": word,
            "lowercase": word.lower(),
            "isCapitalized": word[0].isupper() if word else False,
            "prefix-1": word[:1],
            "prefix-2": word[:2],
            "suffix-1": word[-1:],
            "suffix-2": word[-2:],
            "isDigit": word.isdigit(),
            "prev_word": tokens[idx-1].lower() if idx>0 else "<START>",
            "next_word": tokens[idx+1].lower() if idx<len(tokens)-1 else "<END>",
        }
        feat_vec = vec.transform([feats])
        pred = clf.predict(feat_vec)[0]
        tagged.append({"word": word, "tag": le.inverse_transform([pred])[0]})

    return jsonify({"result": tagged})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
