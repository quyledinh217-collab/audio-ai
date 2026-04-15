import os
import numpy as np
import librosa
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            features = extract_features(filepath)
            features = features.reshape(1, -1)

            pred = model.predict(features)[0]
            prediction = le.inverse_transform([pred])[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)