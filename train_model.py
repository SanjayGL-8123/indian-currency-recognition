"""
Train an Indian currency note classifier (MobileNetV2 transfer learning).
Output: currency_model.h5 and labels.txt
Run:    python train_model.py
"""
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)
BATCH = 32

# Kaggle dataset path (downloaded via kagglehub)
KAGGLE_DATA = r"C:\Users\Sanjay G L\.cache\kagglehub\datasets\vishalmane109\indian-currency-note-images-dataset-2020\versions\1\Indian currency dataset v1"

# Fall back to local dataset/ folder if it has images
LOCAL_DATA = "dataset"

DATA_DIR = KAGGLE_DATA if os.path.isdir(KAGGLE_DATA) else LOCAL_DATA

# Use train/ and val/ if both have images, otherwise split one folder 80/20
train_dir = os.path.join(DATA_DIR, "training")
val_dir = os.path.join(DATA_DIR, "validation")

def has_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for _, _, files in os.walk(folder):
        if any(f.lower().endswith(exts) for f in files):
            return True
    return False

if has_images(train_dir) and has_images(val_dir):
    print("Using separate train/ and val/ folders")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=BATCH, shuffle=True)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=IMG_SIZE, batch_size=BATCH, shuffle=False)
else:
    print("Using dataset/train only, splitting 80% train / 20% validation")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH)

class_names = train_ds.class_names
print("Classes:", class_names)
with open("labels.txt", "w") as f:
    f.write("\n".join(class_names))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
])

base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

# NOTE: preprocess_input is INSIDE the model, so the app must NOT apply it again.
inputs = layers.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)
x = preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("Stage 1: training top layers...")
model.fit(train_ds, validation_data=val_ds, epochs=8)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("Stage 2: fine-tuning...")
model.fit(train_ds, validation_data=val_ds, epochs=6)

model.save("currency_model.h5")
print("Done. Saved currency_model.h5 and labels.txt")
