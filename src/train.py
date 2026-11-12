"""
Automated Multi-Stage Progressive Transfer Learning Training Pipeline
Trains VGG19 Brain Tumor Detection Architecture, computes clinical validation
benchmarks (Accuracy, Sensitivity, Specificity, F1, ROC-AUC), and persists weights.
"""
import os
import json
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.preprocessing import crop_brain_tumor


def augment_image(img: np.ndarray) -> list:
    """Produces synthetic clinical variants (flips, slight rotations) for training."""
    augmented = [img]
    # Horizontal flip
    augmented.append(cv2.flip(img, 1))
    # Vertical flip
    augmented.append(cv2.flip(img, 0))
    # Slight rotations
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    for angle in [10, -10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)
    return augmented


def load_dataset(dataset_dir: str = "dataset", target_size=(120, 120)):
    """Loads and crops axial MRIs from dataset directory."""
    images = []
    labels = []

    classes = [("no", 0), ("yes", 1)]
    for subfolder, label_val in classes:
        folder_path = os.path.join(dataset_dir, subfolder)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Missing folder: {folder_path}")

        files = sorted(os.listdir(folder_path))
        for fname in files:
            p = os.path.join(folder_path, fname)
            img = cv2.imread(p)
            if img is None:
                continue
            cropped = crop_brain_tumor(img)
            resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_CUBIC)
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            images.append(resized_rgb)
            labels.append(label_val)

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)


def train_model(
    dataset_dir: str = "dataset",
    output_dir: str = "model_weights",
    epochs_stage1: int = 12,
    epochs_stage2: int = 8,
    random_state: int = 42
):
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "vgg19_model_03.weights.h5")
    legacy_weights_path = os.path.join(output_dir, "vgg19_model_03.h5")
    keras_model_path = os.path.join(output_dir, "vgg19_model_03.keras")
    metrics_path = os.path.join(output_dir, "metrics.json")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")

    print("[1/5] Loading and morphologically preprocessing dataset...")
    X_raw, y_raw = load_dataset(dataset_dir)
    print(f"Total raw cohort: {len(X_raw)} scans (Tumor: {sum(y_raw == 1)}, Healthy: {sum(y_raw == 0)})")

    # Stratified 80/10/10 split to avoid data leakage
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y_raw, test_size=0.10, stratify=y_raw, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1111, stratify=y_train_val, random_state=random_state
    )

    print(f"Partitions: Train={len(X_train)}, Val={len(X_val)}, Holdout Test={len(X_test)}")

    # Apply data augmentation ONLY to training partition
    print("[2/5] Synthesizing balanced augmentation on train split...")
    X_train_aug = []
    y_train_aug = []
    for img, lbl in zip(X_train, y_train):
        variants = augment_image(img)
        for var in variants:
            X_train_aug.append(var)
            y_train_aug.append(lbl)

    X_train_aug = np.array(X_train_aug, dtype=np.float32) / 255.0
    y_train_aug = tf.keras.utils.to_categorical(np.array(y_train_aug), num_classes=2)

    X_val_norm = np.array(X_val, dtype=np.float32) / 255.0
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=2)

    X_test_norm = np.array(X_test, dtype=np.float32) / 255.0
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=2)

    print(f"Augmented train shape: {X_train_aug.shape}")

    # Build VGG19 architecture
    print("[3/5] Building VGG19 Deep Transfer Model...")
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(120, 120, 3))
    
    # Stage 1: Freeze all conv layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten(name="flatten_head")(x)
    x = Dense(4608, activation='relu', name="dense_proj_4608")(x)
    x = Dropout(0.20, name="dropout_regularizer")(x)
    x = Dense(1152, activation='relu', name="dense_latent_1152")(x)
    output = Dense(2, activation='softmax', name="classification_softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    # Stage 1: Train classification head
    print("[4/5] Stage 1 Training: Classification Head Warm-up...")
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, save_weights_only=True)

    model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val_norm, y_val_cat),
        epochs=epochs_stage1,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # Stage 2: Unfreeze Block 5 for domain adaptation
    print("[5/5] Stage 2 Training: Block 5 Domain Adaptation Fine-Tuning...")
    for layer in base_model.layers:
        if "block5" in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=2e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val_norm, y_val_cat),
        epochs=epochs_stage2,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # Load best weights
    model.load_weights(weights_path)
    # Save full model and legacy alias
    model.save(keras_model_path)
    import shutil
    shutil.copyfile(weights_path, legacy_weights_path)
    print(f"[INFO] Best model weights saved to: {weights_path} and {legacy_weights_path}")

    # Evaluate on Holdout Test Partition
    print("[INFO] Evaluating on unseen holdout test distribution...")
    y_pred_probs = model.predict(X_test_norm, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred_classes)
    prec = precision_score(y_test, y_pred_classes, zero_division=0)
    rec = recall_score(y_test, y_pred_classes, zero_division=0)
    f1 = f1_score(y_test, y_pred_classes, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_probs[:, 1])
    cm = confusion_matrix(y_test, y_pred_classes)

    print("\n" + "="*50)
    print("      HOLDOUT TEST EVALUATION BENCHMARKS")
    print("="*50)
    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"Precision:       {prec * 100:.2f}%")
    print(f"Recall (Sens.):  {rec * 100:.2f}%")
    print(f"F1-Score:        {f1:.4f}")
    print(f"ROC-AUC:         {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("="*50)

    # Save metrics JSON
    metrics_data = {
        "dataset_total": int(len(X_raw)),
        "train_samples": int(len(X_train_aug)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=['No Tumor', 'Tumor'],
        yticklabels=['No Tumor', 'Tumor'],
        title='Confusion Matrix (Holdout Test Cohort)',
        ylabel='True Clinical Label',
        xlabel='Predicted Label'
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black",
                    fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix plotted at: {cm_path}")

    return model, metrics_data


if __name__ == "__main__":
    train_model(epochs_stage1=12, epochs_stage2=8)
