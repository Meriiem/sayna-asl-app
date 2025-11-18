# =============================================================
# 1. SETUP AND IMPORTS
# =============================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import glob
import time
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
# VISUALIZATION IMPORTS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support,  accuracy_score

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Dropout, GlobalAveragePooling2D, GlobalAveragePooling1D, Lambda, Bidirectional, LSTM, Activation
from tensorflow.keras.applications import (
    MobileNetV2,
    EfficientNetB0,
    VGG16,
    DenseNet121,
    InceptionV3,
    ResNet50
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K
import gc
from tqdm.auto import tqdm

tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("TensorFlow Version:", tf.__version__)
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# =============================================================
# 2. CONFIGURATION
# =============================================================
CNN_MODELS_TO_TEST = ['MobileNetV2', 'EfficientNetB0', 'DenseNet121', 'InceptionV3', 'ResNet50']

# Paths
CLASS_DATASET_PATH = "MLR511-ArabicSignLanguage-Dataset-MP4"
KAGGLE_ARSL_PATH = "arslvideodataset"
KAGGLE_20_WORDS_PATH = "asl-20-words-dataset"

# Hyperparameters
IMG_SIZE = (224, 224)
NUM_SEGMENTS = 3
FRAMES_PER_SEGMENT = 25
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
EPOCHS = 50
LOSO_EPOCHS = 20
KAGGLE_EPOCHS = 20

# Classes
CLASS_NAMES = ["G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08", "G09", "G10"]
NUM_CLASSES = 10
LEFT_HANDED_USERS = ['user01', 'user02']
LOSO_TEST_USERS = ['user01', 'user08', 'user11']

# =============================================================
# 3. PREPROCESSING (SAD)
# =============================================================
def extract_sad_features(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((NUM_SEGMENTS, *IMG_SIZE, 3), dtype=np.float32)

    all_frames_gray = []
    normalized_path = os.path.normpath(video_path).split(os.sep)
    is_left_handed = any(part in LEFT_HANDED_USERS for part in normalized_path)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if is_left_handed: frame = cv2.flip(frame, 1)
            frame_resized = cv2.resize(frame, IMG_SIZE)
            all_frames_gray.append(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()

    num_frames = len(all_frames_gray)
    if num_frames < 2:
        return np.zeros((NUM_SEGMENTS, *IMG_SIZE, 3), dtype=np.float32)

    segment_length = num_frames // 2
    overlap = segment_length // 2
    if segment_length < 2:
        return np.array([np.zeros((*IMG_SIZE, 3), dtype=np.float32)] * NUM_SEGMENTS)

    segments = [
        all_frames_gray[0:segment_length],
        all_frames_gray[overlap:segment_length + overlap],
        all_frames_gray[num_frames - segment_length:num_frames],
    ]

    sad_sequence = []
    for segment_frames in segments:
        if len(segment_frames) < 2:
            sad_image_final = np.zeros(IMG_SIZE, dtype=np.float32)
        else:
            indices = np.linspace(0, len(segment_frames) - 1, min(FRAMES_PER_SEGMENT, len(segment_frames)), dtype=int)
            subsampled_frames = [segment_frames[i] for i in indices]

            diff_images = []
            for i in range(len(subsampled_frames) - 1):
                frame1 = subsampled_frames[i].astype(np.float32)
                frame2 = subsampled_frames[i + 1].astype(np.float32)
                diff = cv2.absdiff(frame1, frame2)
                diff_images.append(diff)

            if len(diff_images) > 0:
                all_diffs = np.stack(diff_images, axis=0)
                threshold = np.percentile(all_diffs, 98.0)
                filtered_diffs = [np.where(d < threshold, 0, d) for d in diff_images]
                sad_image = np.sum(filtered_diffs, axis=0)
                sad_image_final = sad_image / (np.max(sad_image) + 1e-6)
            else:
                sad_image_final = np.zeros(IMG_SIZE, dtype=np.float32)

        sad_image_rgb = cv2.cvtColor(sad_image_final.astype(np.float32), cv2.COLOR_GRAY2RGB)
        sad_sequence.append(sad_image_rgb)

    return np.array(sad_sequence, dtype=np.float32)

# =============================================================
# 4. DATA LOADING & GENERATOR
# =============================================================
def is_video_valid(video_path, verbose=False):
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return False
        ret, _ = cap.read()
        return ret
    except Exception:
        return False
    finally:
        if cap: cap.release()

def load_class_dataset(dataset_path, class_names=None):
    print(f"\nLoading class dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"ERROR: Path not found: {dataset_path}")
        return [], [], None

    video_paths, labels = [], []
    search_patterns = [
        os.path.join(dataset_path, "user*", "G*", "*.mp4"),
        os.path.join(dataset_path, "*", "*.mp4"),
    ]

    found_videos = sorted([p for pattern in search_patterns for p in glob.glob(pattern)])

    for vp in tqdm(found_videos, desc="Checking Class Videos"):
        if is_video_valid(vp):
            try:
                label = os.path.basename(os.path.dirname(vp))
                if class_names and label not in class_names: continue
                video_paths.append(vp)
                labels.append(label)
            except Exception:
                continue

    if not video_paths: return [], [], None

    encoder = LabelEncoder()
    if class_names: encoder.fit(class_names)
    else: encoder.fit(sorted(list(set(labels))))

    labels_encoded = encoder.transform(labels)
    return video_paths, labels_encoded, encoder

def load_kaggle_arsl_dataset(dataset_path):
    print(f"\nLoading Kaggle ARSL from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print("ERROR: Path not found."); return [], [], [], [], None

    train_paths, train_labels_str = [], []
    val_paths, val_labels_str = [], []

    # Load Train
    train_split_path = os.path.join(dataset_path, 'train')
    if os.path.exists(train_split_path):
        for class_name in tqdm(os.listdir(train_split_path), desc="Loading ARSL Train"):
            class_path = os.path.join(train_split_path, class_name)
            if os.path.isdir(class_path):
                for vp in glob.glob(os.path.join(class_path, "*.mp4")):
                    if is_video_valid(vp):
                        train_paths.append(vp)
                        train_labels_str.append(class_name)

    # Load Val
    val_split_path = os.path.join(dataset_path, 'val')
    if os.path.exists(val_split_path):
        for class_name in tqdm(os.listdir(val_split_path), desc="Loading ARSL Val"):
            class_path = os.path.join(val_split_path, class_name)
            if os.path.isdir(class_path):
                for vp in glob.glob(os.path.join(class_path, "*.mp4")):
                    if is_video_valid(vp):
                        val_paths.append(vp)
                        val_labels_str.append(class_name)

    encoder = LabelEncoder()
    train_labels_encoded = encoder.fit_transform(train_labels_str)

    val_labels_encoded = []
    val_paths_filtered = []
    for i, label in enumerate(val_labels_str):
        if label in encoder.classes_:
            val_labels_encoded.append(encoder.transform([label])[0])
            val_paths_filtered.append(val_paths[i])

    return train_paths, train_labels_encoded, val_paths_filtered, val_labels_encoded, encoder

def load_kaggle_20_words_dataset(dataset_path):
    print(f"\nLoading Kaggle 20 Words from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print("ERROR: Path not found."); return [], []

    video_paths, labels_str = [], []

    for class_name in tqdm(os.listdir(dataset_path), desc="Loading 20 Words"):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for vp in glob.glob(os.path.join(class_path, "*.mp4")):
                if is_video_valid(vp):
                    video_paths.append(vp)
                    labels_str.append(class_name)

    return video_paths, labels_str

def data_generator(video_paths, labels_one_hot, batch_size, shuffle=True):
    num_samples = len(video_paths)
    indices = np.arange(num_samples)
    while True:
        if shuffle: np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            batch_sad_seq = np.zeros((len(batch_indices), NUM_SEGMENTS, *IMG_SIZE, 3), dtype=np.float32)
            batch_labels = labels_one_hot[batch_indices]

            for i, idx in enumerate(batch_indices):
                batch_sad_seq[i] = extract_sad_features(video_paths[idx])

            yield batch_sad_seq, batch_labels

def preload_dataset(video_paths, labels_one_hot):
    print(f"\n[INFO] Pre-loading {len(video_paths)} videos into RAM to speed up training...")

    # Prepare empty arrays
    # Shape: (Total Videos, 3 Segments, 224, 224, 3)
    X_data = np.zeros((len(video_paths), NUM_SEGMENTS, *IMG_SIZE, 3), dtype=np.float16)
    y_data = labels_one_hot

    for i, path in enumerate(tqdm(video_paths, desc="Extracting Features Once")):
        # This runs SAD logic ONCE per video
        X_data[i] = extract_sad_features(path)

    print(f"[INFO] Dataset loaded. Shape: {X_data.shape}")
    print(f"[INFO] Memory usage: {X_data.nbytes / (1024*1024):.2f} MB")
    return X_data, y_data

# =============================================================
# 5. MODEL ARCHITECTURE (FIXED FOR SERIALIZATION)
# =============================================================
def create_cnn_model(cnn_backbone_name, num_classes):
    model_map = {
        'MobileNetV2': MobileNetV2, 'EfficientNetB0': EfficientNetB0,'DenseNet121': DenseNet121,
        'InceptionV3': InceptionV3, 'ResNet50': ResNet50,
    }

    base_model = model_map[cnn_backbone_name](include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    base_model.trainable = True

    video_input = Input(shape=(NUM_SEGMENTS, *IMG_SIZE, 3), name="video_input")
    cnn_features = TimeDistributed(base_model, name="cnn_feature_extractor")(video_input)

    pooled_features = TimeDistributed(GlobalAveragePooling2D(), name="td_global_pool")(cnn_features)
    temporal_pooled = GlobalAveragePooling1D(name="temporal_pool")(pooled_features)

    x = Dropout(0.5)(temporal_pooled)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # 1. Dense layer with linear activation (logits)
    # 2. Separate Activation layer with float32 dtype
    x = Dense(num_classes, name='logits')(x)
    output = Activation('softmax', dtype='float32', name='output')(x)

    return Model(inputs=video_input, outputs=output, name=f"SAYNA_{cnn_backbone_name}")

# =============================================================
# 6. VISUALIZATION UTILS
# =============================================================
def plot_training_history(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Acc')
    plt.plot(epochs, val_acc, 'r-', label='Validation Acc')
    plt.title(f'{model_name} - Accuracy')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend(); plt.grid(True)

    filename = f"{model_name}_training_curves.png"
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    filename = f"{model_name}_confusion_matrix.png"
    plt.savefig(filename)
    plt.close()

# =============================================================
# 7. EVALUATION FUNCTIONS
# =============================================================
def run_full_dataset_training(cnn_model_name, X_all, y_all):
    print(f"\n>>> TRAINING ON FULL DATASET: {cnn_model_name} <<<")
    results = {'model_name': cnn_model_name}
    K.clear_session(); gc.collect()

    # Stratified split on the PRE-LOADED features
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # Create model
    model = create_cnn_model(cnn_model_name, NUM_CLASSES)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    results['params'] = model.count_params()

    start_train = time.time()
    hist = model.fit(
        x=X_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)],
        verbose=1
    )
    end_train = time.time()
    results['training_time_sec'] = end_train - start_train

    plot_training_history(hist, cnn_model_name)

    val_acc = max(hist.history.get('val_accuracy', [0]))
    results['mlr511_val_accuracy'] = val_acc

    model_path = f"best_model_{cnn_model_name}.h5"
    model.save(model_path)
    results['model_path'] = model_path
    results['model_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)

    # Inference on validation set (using the pre-loaded X_val)
    start_inf = time.time()
    val_preds = model.predict(X_val, verbose=0)
    end_inf = time.time()

    results['inference_time_ms'] = ((end_inf - start_inf) / len(X_val)) * 1000
    val_pred_labels = np.argmax(val_preds, axis=1)
    val_true_labels = np.argmax(y_val, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(val_true_labels, val_pred_labels, average='macro', zero_division=0)
    results['precision'] = precision
    results['recall'] = recall
    results['f1_score'] = f1

    plot_confusion_matrix(val_true_labels, val_pred_labels, CLASS_NAMES, cnn_model_name)
    return results

# =============================================================
# LOSO FUNCTION (Side-by-Side Plots + Timing Table)
# =============================================================
def run_loso_for_best_model(cnn_model_name, X_all, y_all, all_video_paths):
    print(f"\n>>> RUNNING LOSO FOR WINNER: {cnn_model_name} <<<")

    loso_stats = []
    confusion_matrices = []
    users_processed = []

    paths_arr = np.array(all_video_paths)

    for test_user in tqdm(LOSO_TEST_USERS, desc=f"LOSO ({cnn_model_name})"):
        K.clear_session(); gc.collect()

        is_test_user = np.array([test_user in p for p in paths_arr])
        X_train = X_all[~is_test_user]
        y_train = y_all[~is_test_user]
        X_test  = X_all[is_test_user]
        y_test  = y_all[is_test_user]

        if len(X_test) == 0: continue

        model = create_cnn_model(cnn_model_name, NUM_CLASSES)
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        start_train = time.time()
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=LOSO_EPOCHS, verbose=0)
        end_train = time.time()
        train_time = end_train - start_train

        start_inf = time.time()
        pred = model.predict(X_test, verbose=0)
        end_inf = time.time()

        # Calculate average inference time per sample (ms)
        inf_time_ms = ((end_inf - start_inf) / len(X_test)) * 1000

        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(pred, axis=1)
        acc = accuracy_score(y_true, y_pred)

        print(f"  User {test_user}: {acc*100:.2f}% (Train: {train_time:.1f}s)")

        loso_stats.append({
            'User': test_user,
            'Accuracy': acc,
            'Training Time (s)': train_time,
            'Inference Time (ms/sample)': inf_time_ms
        })
        confusion_matrices.append((y_true, y_pred))
        users_processed.append(test_user)

    # --- VISUALIZATION: 3 CONFUSION MATRICES IN 1 ROW ---
    if confusion_matrices:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'LOSO Confusion Matrices - {cnn_model_name}', fontsize=16, fontweight='bold')

        for i, (y_true, y_pred) in enumerate(confusion_matrices):
            ax = axes[i] if len(confusion_matrices) > 1 else axes
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)

            user_name = users_processed[i]
            user_acc = loso_stats[i]['Accuracy']
            ax.set_title(f"User: {user_name}\nAcc: {user_acc*100:.1f}%")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        plt.tight_layout()
        plt.show()

    print("LOSO PERFORMANCE SUMMARY")

    df_loso = pd.DataFrame(loso_stats)
    if not df_loso.empty:
        df_display = df_loso.copy()
        df_display['Accuracy'] = df_display['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
        df_display['Training Time (s)'] = df_display['Training Time (s)'].apply(lambda x: f"{x:.1f}")
        df_display['Inference Time (ms/sample)'] = df_display['Inference Time (ms/sample)'].apply(lambda x: f"{x:.2f}")

        print(df_display.to_string(index=False))

        avg_acc = df_loso['Accuracy'].mean()
        print(f"\nAverage LOSO Accuracy: {avg_acc*100:.2f}%")
    else:
        print("No results to display.")

# =============================================================
# KAGGLE FUNCTION 
# =============================================================
def run_kaggle_for_best_model(cnn_model_name):
    print(f"\n>>> RUNNING EXTERNAL DATASET EVAL FOR WINNER: {cnn_model_name} <<<")

    kaggle_stats = []
    confusion_data = [] 

    # ---------------------------------------------------------
    # 1. KAGGLE ARSL
    # ---------------------------------------------------------
    print(f"\n--- Processing Kaggle ARSL ---")
    K.clear_session(); gc.collect()
    tr_paths, tr_lbls_enc, val_paths, val_lbls_enc, encoder = load_kaggle_arsl_dataset(KAGGLE_ARSL_PATH)

    if tr_paths:
        num_k_classes = len(encoder.classes_)
        tr_lbls_oh = to_categorical(tr_lbls_enc, num_classes=num_k_classes)
        val_lbls_oh = to_categorical(val_lbls_enc, num_classes=num_k_classes)

        tr_gen = data_generator(tr_paths, tr_lbls_oh, BATCH_SIZE, shuffle=True)

        model = create_cnn_model(cnn_model_name, num_k_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

        start_train = time.time()
        model.fit(tr_gen, steps_per_epoch=len(tr_paths)//BATCH_SIZE, epochs=KAGGLE_EPOCHS, verbose=1)
        train_time = time.time() - start_train

        y_true, y_pred = [], []
        inf_times = []

        print("Running Inference on ARSL Val Set...")
        for i in tqdm(range(len(val_paths)), desc="Testing ARSL"):
            sad = extract_sad_features(val_paths[i])

            t0 = time.time()
            pred = model.predict(np.expand_dims(sad, axis=0), verbose=0)
            t1 = time.time()
            inf_times.append(t1 - t0)

            y_true.append(np.argmax(val_lbls_oh[i]))
            y_pred.append(np.argmax(pred))

        avg_inf_ms = (sum(inf_times) / len(inf_times)) * 1000
        acc = accuracy_score(y_true, y_pred)

        kaggle_stats.append({
            'Dataset': 'Kaggle ARSL',
            'Accuracy': acc,
            'Training Time (s)': train_time,
            'Inference Time (ms/sample)': avg_inf_ms
        })
        confusion_data.append((y_true, y_pred, encoder.classes_, "Kaggle ARSL"))

    # ---------------------------------------------------------
    # 2. KAGGLE 20 WORDS
    # ---------------------------------------------------------
    print(f"\n--- Processing Kaggle 20 Words ---")
    K.clear_session(); gc.collect()
    paths, lbls_str = load_kaggle_20_words_dataset(KAGGLE_20_WORDS_PATH)

    if paths:
        tr_p, val_p, tr_l, val_l = train_test_split(paths, lbls_str, test_size=0.2, stratify=lbls_str, random_state=42)

        encoder_20 = LabelEncoder()
        tr_l_enc = encoder_20.fit_transform(tr_l)
        val_l_enc = encoder_20.transform(val_l)
        num_k_classes = len(encoder_20.classes_)

        tr_l_oh = to_categorical(tr_l_enc, num_classes=num_k_classes)
        val_l_oh = to_categorical(val_l_enc, num_classes=num_k_classes)

        tr_gen = data_generator(tr_p, tr_l_oh, BATCH_SIZE, shuffle=True)

        model = create_cnn_model(cnn_model_name, num_k_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

        start_train = time.time()
        model.fit(tr_gen, steps_per_epoch=len(tr_p)//BATCH_SIZE, epochs=KAGGLE_EPOCHS, verbose=1)
        train_time = time.time() - start_train

        y_true, y_pred = [], []
        inf_times = []

        print("Running Inference on 20 Words Val Set...")
        for i in tqdm(range(len(val_p)), desc="Testing 20 Words"):
            sad = extract_sad_features(val_p[i])

            t0 = time.time()
            pred = model.predict(np.expand_dims(sad, axis=0), verbose=0)
            t1 = time.time()
            inf_times.append(t1 - t0)

            y_true.append(np.argmax(val_l_oh[i]))
            y_pred.append(np.argmax(pred))

        avg_inf_ms = (sum(inf_times) / len(inf_times)) * 1000
        acc = accuracy_score(y_true, y_pred)

        kaggle_stats.append({
            'Dataset': 'Kaggle 20 Words',
            'Accuracy': acc,
            'Training Time (s)': train_time,
            'Inference Time (ms/sample)': avg_inf_ms
        })
        confusion_data.append((y_true, y_pred, encoder_20.classes_, "Kaggle 20 Words"))

    # ---------------------------------------------------------
    # 3. VISUALIZATION 
    # ---------------------------------------------------------
    if confusion_data:
        cols = len(confusion_data)
        fig, axes = plt.subplots(1, cols, figsize=(8 * cols, 8))
        if cols == 1: axes = [axes] # Handle single plot case

        fig.suptitle(f'External Datasets Performance - {cnn_model_name}', fontsize=16, fontweight='bold')

        for i, (yt, yp, classes, title) in enumerate(confusion_data):
            ax = axes[i]
            cm = confusion_matrix(yt, yp)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=classes, yticklabels=classes)

            acc_val = kaggle_stats[i]['Accuracy']
            ax.set_title(f"{title}\nAcc: {acc_val*100:.2f}%")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    print("EXTERNAL DATASETS PERFORMANCE SUMMARY")

    if kaggle_stats:
        df_kaggle = pd.DataFrame(kaggle_stats)
        df_kaggle['Accuracy'] = df_kaggle['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
        df_kaggle['Training Time (s)'] = df_kaggle['Training Time (s)'].apply(lambda x: f"{x:.1f}")
        df_kaggle['Inference Time (ms/sample)'] = df_kaggle['Inference Time (ms/sample)'].apply(lambda x: f"{x:.2f}")

        print(df_kaggle.to_string(index=False))
    else:
        print("No external datasets were successfully loaded.")

if 'X_all' not in locals():
    all_video_paths, all_labels_encoded, main_encoder = load_class_dataset(CLASS_DATASET_PATH, CLASS_NAMES)
    if not all_video_paths: raise SystemExit("No videos found.")
    all_labels_one_hot = to_categorical(all_labels_encoded, num_classes=NUM_CLASSES)
    X_all, y_all = preload_dataset(all_video_paths, all_labels_one_hot)

print("\nPHASE 1: COMPARING ALL MODELS ON FULL DATASET\n")

CSV_RESULTS_FILE = "model_metrics_summary.csv"
current_session_results = []

for model_name in tqdm(CNN_MODELS_TO_TEST, desc="Training Models"):
    print(f"\n--- Starting {model_name} ---")

    # 1. Run Training
    res = run_full_dataset_training(model_name, X_all, y_all)

    # This ensures data is saved even if the kernel crashes on the next model
    df_res = pd.DataFrame([res])
    header = not os.path.exists(CSV_RESULTS_FILE) 
    df_res.to_csv(CSV_RESULTS_FILE, mode='a', header=header, index=False)
    print(f"[Saved] Metrics for {model_name} saved to {CSV_RESULTS_FILE}")

    print(f"[Cleanup] Clearing memory after {model_name}...")
    K.clear_session()
    gc.collect()

    current_session_results.append(res)



print("\n" + "="*80 + "\nFINAL METRICS SUMMARY (From CSV)\n" + "="*80)

if os.path.exists(CSV_RESULTS_FILE):
    full_results_df = pd.read_csv(CSV_RESULTS_FILE)
else:
    full_results_df = pd.DataFrame(current_session_results)

if full_results_df.empty:
    raise SystemExit("No training results found. Cannot proceed.")

disp_df = full_results_df.copy()
disp_df = disp_df.drop_duplicates(subset=['model_name'], keep='last') # Keep latest run if duplicates
disp_df['mlr511_val_accuracy'] = disp_df['mlr511_val_accuracy'].apply(lambda x: f"{x*100:.2f}%")
disp_df['params'] = disp_df['params'].apply(lambda x: f"{x/1e6:.2f}M")
disp_df['training_time_sec'] = disp_df['training_time_sec'].apply(lambda x: f"{x:.1f}s")
disp_df['inference_time_ms'] = disp_df['inference_time_ms'].apply(lambda x: f"{x:.2f}ms")
disp_df['f1_score'] = disp_df['f1_score'].apply(lambda x: f"{x:.4f}")
disp_df['model_size_mb'] = disp_df['model_size_mb'].apply(lambda x: f"{x:.2f} MB")

cols_to_show = ['model_name', 'params', 'model_size_mb', 'mlr511_val_accuracy', 'f1_score', 'training_time_sec', 'inference_time_ms']
print(disp_df[cols_to_show].to_string(index=False))

best_row_idx = full_results_df['mlr511_val_accuracy'].idxmax()
best_model_name = full_results_df.loc[best_row_idx, 'model_name']
best_model_path = full_results_df.loc[best_row_idx, 'model_path']
best_accuracy = full_results_df.loc[best_row_idx, 'mlr511_val_accuracy']

print(f"\n" + "#"*60 + f"\nWINNER IS: {best_model_name} ({best_accuracy*100:.2f}%)\n" + "#"*60)

print(f"\nCleaning up storage... Keeping only {best_model_name}.")


for idx, row in full_results_df.iterrows():
    m_name = row['model_name']
    m_path = row['model_path']

    if m_name != best_model_name:
        if os.path.exists(m_path):
            os.remove(m_path)
            print(f"Deleted loser: {m_path}")
        else:
            print(f"File already gone: {m_path}")


best_model_path = f"best_model_{best_model_name}.h5"
tflite_model_path = "best_model_quantized.tflite"

print(f"\n" + "="*60 + "\nPHASE 4: QUANTIZATION ON WINNER\n" + "="*60)
print(f"Quantizing model from: {best_model_path}")

X_train_q, X_val_q, y_train_q, y_val_q = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

h5_size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
print(f"Original H5 Size: {h5_size_mb:.2f} MB")


model_to_quant = tf.keras.models.load_model(best_model_path, safe_mode=False)

def rep_data_gen():
    for i in range(min(100, len(X_train_q))):
        # Model expects shape (1, segments, h, w, 3)
        input_data = X_train_q[i].astype(np.float32)
        yield [np.expand_dims(input_data, axis=0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model_to_quant)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Converting model (this may take a moment)...")
tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

tflite_size_mb = os.path.getsize(tflite_model_path) / (1024 * 1024)
print(f"Quantized TFLite Size: {tflite_size_mb:.2f} MB")

def evaluate_tflite_model(tflite_path, X_test, y_test):
    print("Running inference on TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details['quantization']
    input_dtype = input_details['dtype'] # Should be int8

    y_pred_indices = []
    y_true_indices = np.argmax(y_test, axis=1)

    for i in tqdm(range(len(X_test)), desc="TFLite Inference"):
        raw_input = X_test[i].astype(np.float32)

        input_tensor = (raw_input / input_scale) + input_zero_point
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(input_dtype)

        interpreter.set_tensor(input_details['index'], input_tensor)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details['index'])

        y_pred_indices.append(np.argmax(output))

    acc = accuracy_score(y_true_indices, y_pred_indices)
    return acc

tflite_acc = evaluate_tflite_model(tflite_model_path, X_val_q, y_val_q)

original_acc = best_accuracy if 'best_accuracy' in locals() else 0.0

comparison_data = {
    'Metric': ['Model Size (MB)', 'Accuracy (%)', 'Format'],
    'Before Quantization (.h5)': [f"{h5_size_mb:.2f}", f"{original_acc*100:.2f}", "Float32"],
    'After Quantization (.tflite)': [f"{tflite_size_mb:.2f}", f"{tflite_acc*100:.2f}", "Int8"],
    'Change': [f"{h5_size_mb/tflite_size_mb:.1f}x Smaller", f"{(tflite_acc-original_acc)*100:.2f}%", "Converted"]
}

df_compare = pd.DataFrame(comparison_data)

print("\n" + "#"*50)
print("FINAL QUANTIZATION REPORT")
print("#"*50)
print(df_compare.to_string(index=False))
print("#"*50)


print("\n" + "="*60 + "\nPHASE 2: RUNNING LOSO ON WINNER\n" + "="*60)

if 'X_all' not in locals():
    print("Reloading dataset into RAM...")
    all_video_paths, all_labels_encoded, main_encoder = load_class_dataset(CLASS_DATASET_PATH, CLASS_NAMES)
    all_labels_one_hot = to_categorical(all_labels_encoded, num_classes=NUM_CLASSES)
    X_all, y_all = preload_dataset(all_video_paths, all_labels_one_hot)

run_loso_for_best_model(best_model_name, X_all, y_all, all_video_paths)


#  Kaggle on Winner
print("\n" + "="*60 + "\nPHASE 3: RUNNING EXTERNAL DATASETS ON WINNER\n" + "="*60)
run_kaggle_for_best_model(best_model_name)



