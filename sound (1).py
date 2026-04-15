
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


METADATA_PATH = Path(r"D:\sound_project\data\ESC-50-master\ESC-50-master\meta\esc50.csv")
AUDIO_DIR = Path(r"D:\sound_project\data\ESC-50-master\ESC-50-master\audio")
BASE_DIR = Path(__file__).resolve().parent
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
learning_rate = 1e-4 
fine_tune_lr = 1e-5 


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")

    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} samples from {path}")
    return df


def clean_metadata(df: pd.DataFrame, audio_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["filename"] = df["filename"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["fold"] = df["fold"].astype(int)
    df["target"] = df["target"].astype(int)
    df["src_file"] = df["src_file"].astype(int)
    df["esc10"] = df["esc10"].astype(str).str.lower().eq("true")
    df["major_category_id"] = df["target"] // 10
    df["file_path"] = df["filename"].apply(lambda name: audio_dir / name)

    duplicate_rows = int(df.duplicated(subset=["filename"]).sum())
    missing_files = df.loc[~df["file_path"].apply(Path.exists), "filename"].tolist()

    print(f"[INFO] Duplicate filenames: {duplicate_rows}")
    print(f"[INFO] Missing audio files: {len(missing_files)}")

    if duplicate_rows > 0:
        df = df.drop_duplicates(subset=["filename"]).reset_index(drop=True)

    if missing_files:
        raise FileNotFoundError(f"Audio files are missing. Example files: {missing_files[:5]}")

    return df



def load_audio(path: Path) -> tuple[np.ndarray, int]:
    signal, sr = librosa.load(path, sr=TARGET_SAMPLE_RATE, mono=True)
    return signal, int(sr)




def fix_length(signal: np.ndarray, sr: int) -> np.ndarray:
    target_len = int(sr * TARGET_DURATION)
    if len(signal) < target_len:
        return np.pad(signal, (0, target_len - len(signal)), mode="constant")
    return signal[:target_len]


def normalize(signal: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(signal))
    return signal if max_val == 0 else signal / max_val


def preprocess_waveform(path: Path) -> tuple[np.ndarray, int]:
    signal, sr = load_audio(path)
    signal = fix_length(signal, sr)
    signal = normalize(signal)
    return signal, sr


def extract_log_mel(signal, sr):
    mel = librosa.feature.melspectrogram(
        y=signal, 
        sr = sr,
        n_mels = 128,
        fmax = 8000,
        hop_length=512,
        n_fft= 2048
    )
    log_mel = librosa.power_to_db(mel)

    # chuan hoa manh hon 
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)

    return log_mel.astype(np.float32)


def extract_classical_features(signal: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(signal)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=signal)

    return np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            delta.mean(axis=1),
            delta.std(axis=1),
            delta2.mean(axis=1),
            delta2.std(axis=1),
            [zcr.mean(), zcr.std()],
            [centroid.mean(), centroid.std()],
            [bandwidth.mean(), bandwidth.std()],
            [rolloff.mean(), rolloff.std()],
            [flatness.mean(), flatness.std()],
        ]
    ).astype(np.float32)


def collect_audio_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        if idx % 250 == 0 or idx == total:
            print(f"[INFO] EDA audio scan {idx}/{total}")

        rows.append(
            {
                "filename": row.filename,
                "sample_rate": librosa.get_samplerate(path=str(row.file_path)),
                "duration_seconds": librosa.get_duration(path=str(row.file_path)),
            }
        )

    return df.merge(pd.DataFrame(rows), on="filename", how="left")


def create_eda_outputs(df: pd.DataFrame) -> None:
    summary_df = pd.DataFrame(
        [
            {
                "samples": len(df),
                "classes": df["category"].nunique(),
                "folds": df["fold"].nunique(),
                "esc10_samples": int(df["esc10"].sum()),
                "duration_min": df["duration_seconds"].min(),
                "duration_mean": df["duration_seconds"].mean(),
                "duration_max": df["duration_seconds"].max(),
            }
        ]
    )
    summary_df.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)

    class_counts = (
        df["category"]
        .value_counts()
        .sort_values(ascending=False)
        .rename_axis("category")
        .reset_index(name="count")
    )
    class_counts.to_csv(OUTPUT_DIR / "class_counts.csv", index=False)
    pd.crosstab(df["fold"], df["category"]).to_csv(OUTPUT_DIR / "fold_class_counts.csv")
    df["sample_rate"].value_counts().sort_index().to_csv(
        OUTPUT_DIR / "sample_rate_counts.csv", header=["count"]
    )

    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x="fold")
    plt.title("Fold Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fold_distribution.png")
    plt.close()

    plt.figure(figsize=(14, 6))
    sns.barplot(data=class_counts, x="count", y="category", color="steelblue")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    durations = df["duration_seconds"].dropna().astype(float)
    sns.histplot(data = df, x ="duration_seconds", bins=20, kde=True) # type : ingore
    plt.title("Duration Distribution")
    plt.xlabel("Seconds")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "duration_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(x=df["sample_rate"].astype(str), color="darkorange")
    plt.title("Sample Rate Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_rate_distribution.png")
    plt.close()

    for example_index in EXAMPLE_ROWS:
        sample = df.iloc[example_index]
        signal, sr = preprocess_waveform(sample["file_path"])
        spec = extract_log_mel(signal, sr)

        plt.figure(figsize=(12, 3))
        librosa.display.waveshow(signal, sr=sr)
        plt.title(f"Waveform: {sample['category']}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"waveform_{example_index}.png")
        plt.close()

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Log-Mel Spectrogram: {sample['category']}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"logmel_{example_index}.png")
        plt.close()

    print(f"[INFO] Saved EDA outputs to {OUTPUT_DIR}")


def build_datasets(df: pd.DataFrame, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_classical_path = OUTPUT_DIR / "X_classical.npy"
    X_cnn_path = OUTPUT_DIR / "X_cnn.npy"
    y_path = OUTPUT_DIR / "y_labels.npy"
    index_path = OUTPUT_DIR / "feature_index.csv"
    

    if use_cache and X_classical_path.exists() and X_cnn_path.exists() and y_path.exists():
        print("[INFO] Loading cached feature arrays")
        X_classical = np.load(X_classical_path)
        X_cnn = np.load(X_cnn_path)
        y = np.load(y_path, allow_pickle=True)

        # Guard: if cached CNN shape is wrong, force a re-build

        if X_cnn.shape[1:] != (224, 224):
            print(f"[WARN] Cached X_cnn shape {X_cnn.shape[1:]} != (224, 224). Rebuilding cache.")
        else: 

            return X_classical, X_cnn, y

    classical_features = []
    cnn_features = []
    labels = []
    total = len(df)

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        if idx % 100 == 0 or idx == total:
            print(f"[INFO] Feature extraction {idx}/{total}")

        signal, sr = preprocess_waveform(Path(str(row.file_path)))


        classical_features.append(extract_classical_features(signal, sr))

        # CNN features
        log_mel = extract_log_mel(signal, sr)


        log_mel_3d = log_mel[...,np.newaxis]

        log_mel_224 = tf.image.resize(log_mel_3d, (224, 224)).numpy() # (224, 224, 1)

        log_mel_224 = log_mel_224.squeeze(axis=-1) # (224, 224)

     

        # augmentation chi cho train (optional: luon apply nhe)

        cnn_features.append(log_mel_224.astype(np.float32))

        labels.append(row.category)


    X_classical = np.vstack(classical_features)
    X_cnn = np.stack(cnn_features) # shape: (N ,224, 224)
    y = np.array(labels)

    np.save(X_classical_path, X_classical)
    np.save(X_cnn_path, X_cnn)
    np.save(y_path, y)
    df[["filename", "category", "fold"]].to_csv(index_path, index=False)

    print(f"[INFO] Classical feature shape: {X_classical.shape}")
    print(f"[INFO] CNN tensor shape: {X_cnn.shape}") #should be (2000, 224, 224)
    return X_classical, X_cnn, y


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_classical_model(
    model_name: str,
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    class_names: list[str],
) -> dict:
    fold_rows = []
    all_predictions = np.empty_like(y, dtype=object)

    for fold in sorted(np.unique(folds)):
        print(f"[INFO] {model_name}: training fold {fold}")
        train_mask = folds != fold
        test_mask = folds == fold

        estimator.fit(X[train_mask], y[train_mask])
        predictions = estimator.predict(X[test_mask])
        all_predictions[test_mask] = predictions

        fold_rows.append(
            {
                "model": model_name,
                "fold": int(fold),
                "accuracy": accuracy_score(y[test_mask], predictions),
                "macro_f1": f1_score(y[test_mask], predictions, average="macro"),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    average_row = pd.DataFrame(
        [
            {
                "model": model_name,
                "fold": "average",
                "accuracy": fold_df["accuracy"].mean(),
                "macro_f1": fold_df["macro_f1"].mean(),
            }
        ]
    )
    fold_df = pd.concat([fold_df, average_row], ignore_index=True)
    fold_df.to_csv(OUTPUT_DIR / f"{model_name.lower()}_fold_results.csv", index=False)

    pd.DataFrame(
        classification_report(y, all_predictions, labels=class_names, output_dict=True)
    ).transpose().to_csv(OUTPUT_DIR / f"{model_name.lower()}_classification_report.csv")

    save_confusion_matrix(
        y,
        all_predictions,
        class_names,
        OUTPUT_DIR / f"{model_name.lower()}_confusion_matrix.png",
        f"{model_name} Confusion Matrix",
    )

    final_estimator = estimator.fit(X, y)
    joblib.dump(final_estimator, MODEL_DIR / f"{model_name.lower()}_model.joblib")

    avg_accuracy = float(fold_df.loc[fold_df["fold"] == "average", "accuracy"].iloc[0])
    avg_macro_f1 = float(fold_df.loc[fold_df["fold"] == "average", "macro_f1"].iloc[0])

    print(f"[INFO] {model_name} average accuracy: {avg_accuracy:.4f}")
    print(f"[INFO] {model_name} average macro F1: {avg_macro_f1:.4f}")
    return {
        "model": model_name,
        "accuracy": avg_accuracy,
        "macro_f1": avg_macro_f1,
        "status": "completed",
    }


def build_svm_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale")),
        ]
    )


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )


def build_cnn_model(input_shape, num_classes):


    base_model = tf.keras.applications.EfficientNetB0(
        include_top = False,
        weights = "imagenet",
        input_shape=(224, 224, 3),
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape = input_shape) # e.g (224, 224, 1) 


    # Resize first
    

    # 1 -> 3 channels 
    x = tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis =-1))(inputs)

    

    # Noramlize to 224 
    

    x = base_model(x, training = False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x= tf.keras.layers.BatchNormalization()(x)


    x = tf.keras.layers.Dense(256, activation= "relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation = "softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer =tf.keras.optimizers.Adam(learning_rate),
        loss = "sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def evaluate_cnn(
    X_cnn: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    class_names: list[str],
) -> dict:
    tf.random.set_seed(RANDOM_STATE)
    y_indices = np.array([class_names.index(label) for label in y], dtype=np.int32)
    X_cnn = X_cnn[..., np.newaxis]
    all_predictions = np.empty_like(y_indices)
    fold_rows = []

    

    for fold in sorted(np.unique(folds)):
        print(f"[INFO] CNN: training fold {fold}")
        train_mask = folds != fold
        test_mask = folds == fold

        X_train, y_train = X_cnn[train_mask], y_indices[train_mask]
        X_val, y_val = X_cnn[test_mask], y_indices[test_mask]

        from sklearn.utils.class_weight import compute_class_weight
        cw_values = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        class_weights = dict(enumerate(cw_values))


        # Phase 1
        model, base_model = build_cnn_model(X_cnn.shape[1:], len(class_names))

        base_model.trainable = False


        callbacks_phase1 = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=7,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=3,
                min_lr=1e-6,
            )
        ]

        def augment_map(x, label):
            x = tf.numpy_function(
                func = lambda s: spec_augment(s.squeeze())[...,np.newaxis].astype(np.float32),
                inp =[x],
                Tout = tf.float32,

            )
            x.set_shape(X_cnn.shape[1:])
            return x, label 
        
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(y_train),seed = RANDOM_STATE)
            .map(augment_map, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)

        )

        val_ds = ( 
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
        


        model.fit(
            train_ds,
            validation_data= val_ds,
            epochs=25,
            shuffle = True,
            callbacks=callbacks_phase1,
            class_weight= class_weights,
        )



        # Phase 2 ~ unfreeze top 20 layers, fine-tune

        base_model.trainable = True
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[-20:]:
            layer.trainable = True
        

        callbacks_phase2 = [
            tf.keras.callbacks.EarlyStopping(
                monitor ='val_loss',
                patience = 7,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-7,
            )
        ]


        model.compile(
            optimizer= tf.keras.optimizers.Adam(5e-6),
            loss = "sparse_categorical_crossentropy",
            metrics = ["accuracy"]
            
        )
            
        model.fit(
            train_ds,
            validation_data = val_ds,
            epochs=20,
            verbose = 1,
            callbacks=callbacks_phase2,
            class_weight= class_weights,
        )


        #Evaluate fold
        probabilities = model.predict(X_cnn[test_mask], verbose=0)
        predictions = probabilities.argmax(axis=1)
        all_predictions[test_mask] = predictions

        fold_rows.append(
            {
                "model": "CNN",
                "fold": int(fold),
                "accuracy": accuracy_score(y_val, predictions),
                "macro_f1": f1_score(y_val, predictions, average="macro"),
            }
        )

    # Aggregate fold results
    fold_df = pd.DataFrame(fold_rows)
    average_row = pd.DataFrame(
        [
            {
                "model": "CNN",
                "fold": "average",
                "accuracy": fold_df["accuracy"].mean(),
                "macro_f1": fold_df["macro_f1"].mean(),
            }
        ]
    )

    
    fold_df = pd.concat([fold_df, average_row], ignore_index=True)
    fold_df.to_csv(OUTPUT_DIR / "cnn_fold_results.csv", index=False)

    predicted_labels = np.array([class_names[i] for i in all_predictions], dtype=object)
    pd.DataFrame(
        classification_report(y, predicted_labels, labels=class_names, output_dict=True)
    ).transpose().to_csv(OUTPUT_DIR / "cnn_classification_report.csv")

    save_confusion_matrix(
        y,
        predicted_labels,
        class_names,
        OUTPUT_DIR / "cnn_confusion_matrix.png",
        "CNN Confusion Matrix",
    )

    
    # Final model: 
    print("[INFO] Training final CNN on full dataset")
    final_model, final_base = build_cnn_model(X_cnn.shape[1:], len(class_names))
    full_ds =(
        tf.data.Dataset.from_tensor_slices((X_cnn, y_indices))
        .shuffle(len(y_indices), seed=RANDOM_STATE)

        .map(augment_map, num_parallel_calls=tf.data.AUTOTUNE)
    )


    # Phase 1 - frozen base
    final_base.trainable = False
    final_model.fit(full_ds, epochs=25, verbose = 1)


    # Phase 2 - unfreeze top 20 
    final_base.trainable = True
    for layer in final_base.layers:
        layer.trainable = False
    for layer in final_base.layers[-20:]:
        layer.trainable = True


    final_model.compile(
        optimizer = tf.keras.optimizers.Adam(5e-6),
        loss = "sparse_categorical_crossentropy",
        metrics =["accuracy"]
    )
    final_model.fit(full_ds, epochs=20, verbose=1)
    final_model.save(MODEL_DIR / "cnn_model.keras")
    print("[INFO] Final CNN model saved.")


    # Return summary
    avg_accuracy = float(fold_df.loc[fold_df["fold"] == "average", "accuracy"].iloc[0])
    avg_macro_f1 = float(fold_df.loc[fold_df["fold"] == "average", "macro_f1"].iloc[0])

    print(f"[INFO] CNN average accuracy: {avg_accuracy:.4f}")
    print(f"[INFO] CNN average macro F1: {avg_macro_f1:.4f}")
    return {
        "model": "CNN",
        "accuracy": avg_accuracy,
        "macro_f1": avg_macro_f1,
        "status": "completed",
    }

def spec_augment(spec):
    spec = spec.copy()

    # Time masking 
    if np.random.rand() < 0.5: 
        t = np.random.randint(0,max(1, spec.shape[1] -20))
        spec[:, t:t+20] = 0
    # Frequency masking 
    if np.random.rand() < 0.5: 
        f = np.random.randint(0,max(1, spec.shape[0] - 10))
        spec[f:f +10,:] = 0
    
    return spec


def main() -> None:
    ensure_output_dirs()

    print("[STEP 1] Load and clean metadata")
    df = load_metadata(METADATA_PATH)
    df = clean_metadata(df, AUDIO_DIR)

    print("[STEP 2] Run EDA")
    df = collect_audio_statistics(df)
    create_eda_outputs(df)

    print("[STEP 3] Build datasets")
    X_classical, X_cnn, y = build_datasets(df, use_cache=True)
    folds = df["fold"].to_numpy()
    class_names = sorted(df["category"].unique().tolist())

    print("[STEP 4] Train SVM baseline")
    svm_result = evaluate_classical_model(
        "SVM",
        build_svm_pipeline(),
        X_classical,
        y,
        folds,
        class_names,
    )

    print("[STEP 5] Train Random Forest")
    rf_result = evaluate_classical_model(
        "RandomForest",
        build_random_forest(),
        X_classical,
        y,
        folds,
        class_names,
    )

    print("[STEP 6] Train CNN")
    cnn_result = evaluate_cnn(X_cnn, y, folds, class_names)

    comparison_df = pd.DataFrame([svm_result, rf_result, cnn_result])
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\n[RESULTS]")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()

