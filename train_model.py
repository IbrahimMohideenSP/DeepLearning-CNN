import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from collections import Counter  # ✅ fix: import Counter

# --------------------
# CONFIG
# --------------------
IMG_SIZE = 224
DATA_DIR = "data/train/color"
MODEL_SAVE_PATH = "anti_spoof_model.h5"

# --------------------
# LOAD IMAGES & LABELS
# --------------------
def load_images(data_dir):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(data_dir, filename)
            label = 1 if "real" in filename.lower() else 0  # 1 = Real, 0 = Fake
            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

print("[INFO] Loading images...")
X, y = load_images(DATA_DIR)

print(f"[INFO] Dataset shape: {X.shape}, Labels shape: {y.shape}")
print("[INFO] Label Distribution:", Counter(y))
for label, count in Counter(y).items():
    print(f"{'REAL' if label == 1 else 'FAKE'}: {count} images ({(count/len(y))*100:.2f}%)")

# --------------------
# BALANCE THE DATASET
# --------------------
real_idx = np.where(y == 1)[0]
fake_idx = np.where(y == 0)[0]
min_len = min(len(real_idx), len(fake_idx))

balanced_idx = np.concatenate([real_idx[:min_len], fake_idx[:min_len]])
X_balanced = X[balanced_idx]
y_balanced = y[balanced_idx]

# Normalize and split
X_balanced = X_balanced.astype("float32") / 255.0
X_train, X_val, y_train, y_val = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
)

# --------------------
# BUILD MODEL
# --------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# --------------------
# TRAIN MODEL
# --------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
]

print("[INFO] Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# --------------------
# PLOT RESULTS
# --------------------
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

print(f"[INFO] Model saved to {anti_spoof_model.h5}")