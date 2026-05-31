from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Rescaling,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np

# -----------------------------------
# SETTINGS
# -----------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# -----------------------------------
# LOAD DATASET
# -----------------------------------
train_dir = "data/Train"

train_ds, val_ds = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="both",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# -----------------------------------
# CLASS NAMES
# -----------------------------------
class_names = train_ds.class_names

print("\nClasses:")
print(class_names)

# -----------------------------------
# PREFETCH (FASTER TRAINING)
# -----------------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -----------------------------------
# DATA AUGMENTATION
# -----------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# -----------------------------------
# BUILD MODEL
# -----------------------------------
model = Sequential([

    # Data augmentation
    data_augmentation,

    # Normalize
    Rescaling(1./255, input_shape=(224, 224, 3)),

    # CNN Layers
    Conv2D(32, 3, activation="relu"),
    MaxPooling2D(),

    Conv2D(64, 3, activation="relu"),
    MaxPooling2D(),

    Conv2D(128, 3, activation="relu"),
    MaxPooling2D(),

    Flatten(),

    Dense(128, activation="relu"),

    Dropout(0.3),

    Dense(len(class_names), activation="softmax")
])

# -----------------------------------
# COMPILE MODEL
# -----------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------------
# EARLY STOPPING
# -----------------------------------
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

# -----------------------------------
# TRAIN MODEL
# -----------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# -----------------------------------
# FINAL ACCURACY
# -----------------------------------
train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]

print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# -----------------------------------
# SAVE MODEL
# -----------------------------------
model.save("model.h5")

print("\n✅ Model saved as model.h5")