import tensorflow as tf
from tensorflow.keras import layers, models

def build_ann_titanic(input_dim, num_classes=2):
    """Builds a simple ANN for Titanic tabular data."""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model