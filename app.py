from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os
import librosa

app = Flask(__name__)

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(path):
    return joblib.load(os.path.join(BASE_DIR, path))

# ===== LOAD KNN (ĐÚNG FILE CỦA BẠN) =====
knn_model = load_model('model/knn_urban_model.pkl')
knn_scaler = load_model('model/knn_urban_scaler.pkl')

print("✅ KNN Urban model loaded")

# ================= FEATURE =================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000, duration=4)
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

# ================= ROUTES =================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    # ===== SAVE FILE =====
    upload_dir = os.path.join(BASE_DIR, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    try:
        # ===== FEATURE =====
        features = extract_features(file_path).reshape(1, -1)

        # ===== KNN =====
        f = knn_scaler.transform(features)
        pred = knn_model.predict(f)[0]
        conf = float(np.max(knn_model.predict_proba(f)[0]))

    except Exception as e:
        return jsonify({'error': str(e)})

    # ===== WAVEFORM =====
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=4)
        waveform = y.tolist()[:2000]
    except:
        waveform = []

    return jsonify({
        'knn_prediction': str(pred),
        'knn_confidence': conf,
        'waveform': waveform
    })

# ================= RUN =================
if __name__ == "__main__":
    print("🚀 Server running (KNN Urban)...")
    app.run(host="0.0.0.0", port=5000, debug=True)