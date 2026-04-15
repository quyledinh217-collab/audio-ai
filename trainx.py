from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ================= CONFIG =================
METADATA_PATH = Path(r"D:\sound_project\data\ESC-50-master\ESC-50-master\meta\esc50.csv")
AUDIO_DIR = Path(r"D:\sound_project\data\ESC-50-master\ESC-50-master\audio")

# ✅ FIX __file__
BASE_DIR = Path().resolve()

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

TARGET_SAMPLE_RATE = 22050
TARGET_DURATION = 5.0
N_MELS = 128
N_MFCC = 20
CNN_EPOCHS = 45 
BATCH_SIZE = 64
EXAMPLE_ROWS = [0, 400, 800]
RANDOM_STATE = 42

# ================= DIR =================
def ensure_output_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

# ================= LOAD =================
def load_metadata(path):
    df = pd.read_csv(path)
    df["file_path"] = df["filename"].apply(lambda x: AUDIO_DIR / x)
    return df

def clean_metadata(df, audio_dir):
    df = df.copy()
    df["file_path"] = df["filename"].apply(lambda x: audio_dir / x)
    return df

# ================= AUDIO =================
def preprocess_waveform(path):
    signal, sr = librosa.load(path, sr=TARGET_SAMPLE_RATE)
    target_len = int(sr * TARGET_DURATION)

    if len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)))
    else:
        signal = signal[:target_len]

    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    return signal, sr

# ================= FEATURE =================
def extract_log_mel(signal, sr):
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.astype(np.float32)

def extract_classical_features(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

# ================= DATASET =================
def build_datasets(df, use_cache=True):
    Xc_path = OUTPUT_DIR / "X_classical.npy"
    Xcnn_path = OUTPUT_DIR / "X_cnn.npy"
    y_path = OUTPUT_DIR / "y.npy"

    if use_cache and Xc_path.exists():
        return np.load(Xc_path), np.load(Xcnn_path), np.load(y_path, allow_pickle=True)

    X_classical, X_cnn, y = [], [], []

    for i, row in df.iterrows():
        if i % 200 == 0:
            print(i)

        signal, sr = preprocess_waveform(row["file_path"])

        # classical
        X_classical.append(extract_classical_features(signal, sr))

        # CNN ✅ FIX SHAPE
        log_mel = extract_log_mel(signal, sr)
        log_mel = np.expand_dims(log_mel, axis=-1)  # (H,W,1)
        log_mel = tf.image.resize(log_mel, (224, 224)).numpy()

        X_cnn.append(log_mel)
        y.append(row["category"])

    X_classical = np.array(X_classical)
    X_cnn = np.array(X_cnn)
    y = np.array(y)

    np.save(Xc_path, X_classical)
    np.save(Xcnn_path, X_cnn)
    np.save(y_path, y)

    return X_classical, X_cnn, y

# ================= MODEL =================
def build_svm_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=10))
    ])

def build_random_forest():
    return RandomForestClassifier(n_estimators=200)

# ================= CNN =================
def build_cnn_model(input_shape, num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224,224,3)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1))(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ================= TRAIN =================
def evaluate_classical_model(name, model, X, y):
    model.fit(X, y)
    joblib.dump(model, MODEL_DIR / f"{name}.pkl")
    return {"model": name, "status": "done"}

def evaluate_cnn(X, y):
    labels = sorted(set(y))
    y_idx = np.array([labels.index(i) for i in y])

    model = build_cnn_model(X.shape[1:], len(labels))

    model.fit(X, y_idx, epochs=CNN_EPOCHS, batch_size=BATCH_SIZE)

    model.save(MODEL_DIR / "cnn_model.keras")
    joblib.dump(labels, MODEL_DIR / "labels.pkl")

    return {"model": "CNN", "status": "done"}

# ================= MAIN =================
def main():
    ensure_output_dirs()

    print("Load data...")
    df = load_metadata(METADATA_PATH)
    df = clean_metadata(df, AUDIO_DIR)

    print("Build dataset...")
    Xc, Xcnn, y = build_datasets(df)

    print("Train SVM...")
    evaluate_classical_model("svm", build_svm_pipeline(), Xc, y)

    print("Train RF...")
    evaluate_classical_model("rf", build_random_forest(), Xc, y)

    print("Train CNN...")
    evaluate_cnn(Xcnn, y)

    print("DONE 🚀")

if __name__ == "__main__":
    main()