import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

print("Building model...")

# exact architecture from training
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

print("Loading weights...")

# your local weights file
model.load_weights("models/retina_weights.weights.h5")

print("Saving full model...")

model.save("models/final_resnet50_model.h5")

print("DONE. Full model saved.")