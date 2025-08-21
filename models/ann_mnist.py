import tensorflow as tf
from tensorflow.keras import layers, models

def build_ann_mnist(input_shape=(784,), num_classes=10):
    """Builds a simple ANN for MNIST (flattened)."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model