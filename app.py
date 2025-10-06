from flask import Flask, request, render_template
import joblib
import os
import sys
from pathlib import Path

current_dir = Path(__file__).parent

project_root = current_dir

# Import clean_text from your preprocessing
sys.path.append(os.path.dirname(__file__))  # ensure src/ is in path
from preprocessing import clean_text  

# Paths
MODEL_PATH = project_root / "data" / "fake_news_model.pkl"
VECTORIZER_PATH = project_root / "data" / "tfidf_vectorizer.pkl"

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

LABEL_MAP = {0: "FAKE NEWS ❌", 1: "REAL NEWS ✅"}

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result, confidence, text = None, None, ""
    
    if request.method == "POST":
        text = request.form["news"]
        text_clean = clean_text(text)
        features = vectorizer.transform([text_clean])
        pred = int(model.predict(features)[0])
        result = LABEL_MAP.get(pred, str(pred))

        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(features)[0][pred])

    return render_template("index.html", result=result, confidence=confidence, text=text)


if __name__ == "__main__":
    app.run(debug=True)
