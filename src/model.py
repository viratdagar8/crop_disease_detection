"""CNN Model Architecture for Crop Disease Detection.
Aligns with PRD Section 12 (Baseline CNN Architecture) and Section 22 (Future Enhancements).
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_baseline_cnn(input_shape=(128, 128, 3), num_classes=38):
    """Builds a robust, explainable baseline Convolutional Neural Network (CNN).
    
    Architecture:
        - Block 1: Conv2D(32) -> BatchNormalization -> ReLU -> MaxPool2D -> Dropout(0.25)
        - Block 2: Conv2D(64) -> BatchNormalization -> ReLU -> MaxPool2D -> Dropout(0.25)
        - Block 3: Conv2D(128) -> BatchNormalization -> ReLU -> MaxPool2D -> Dropout(0.25)
        - Block 4: Conv2D(256) -> BatchNormalization -> ReLU -> MaxPool2D -> Dropout(0.25)
        - Classifier: Flatten -> Dense(256) -> BatchNormalization -> ReLU -> Dropout(0.5) -> Dense(num_classes, Softmax)
    
    This architecture is directly aligned with PRD Section 12, balancing feature extraction 
    depth with regularization to prevent overfitting on leaf textures.
    """
    model = models.Sequential(name="CropDisease_BaselineCNN")

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # Convolutional Block 1 (Low-level edge & color filter detection)
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1"))
    model.add(layers.BatchNormalization(name="bn1"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool1"))
    model.add(layers.Dropout(0.2, name="drop1"))

    # Convolutional Block 2 (Texture & leaf lesion spot detection)
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2"))
    model.add(layers.BatchNormalization(name="bn2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool2"))
    model.add(layers.Dropout(0.2, name="drop2"))

    # Convolutional Block 3 (Pattern & complex symptom detection)
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3"))
    model.add(layers.BatchNormalization(name="bn3"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool3"))
    model.add(layers.Dropout(0.25, name="drop3"))

    # Convolutional Block 4 (High-level disease category representation)
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4"))
    model.add(layers.BatchNormalization(name="bn4"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool4"))
    model.add(layers.Dropout(0.25, name="drop4"))

    # Classification Head
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(256, activation="relu", name="dense1"))
    model.add(layers.BatchNormalization(name="bn_dense"))
    model.add(layers.Dropout(0.5, name="drop_dense"))
    model.add(layers.Dense(num_classes, activation="softmax", name="output_probabilities"))

    return model


def build_mobilenet_transfer_model(input_shape=(128, 128, 3), num_classes=38):
    """Optional Transfer Learning model using MobileNetV2 (PRD Section 12 & 22).
    Lightweight, fast, and highly accurate for deployment.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze base feature extractor

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="CropDisease_MobileNetV2")
    return model
