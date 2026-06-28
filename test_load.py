import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

print("Building model...")

base_model = ResNet50(
    weights=None,
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = True

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(5, activation="softmax")
])

# IMPORTANT
model.build((None, 224, 224, 3))

print("Loading weights...")

model.load_weights("models/retina_weights.weights.h5")

print("SUCCESS")