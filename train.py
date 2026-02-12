import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import os

# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load CIFAR-10 dataset

def load_data(batch_size=64):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    AUTOTUNE = tf.data.AUTOTUNE

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_rotation(image, 0.1)  # 0.1 ~ 10% of 2pi
        image = tf.image.random_zoom(image, (0.9, 1.1))
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=5000)
    train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, test_ds

# Build a simple CNN model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    batch_size = 64
    train_ds, test_ds = load_data(batch_size)
    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_ds, epochs=10, validation_data=test_ds)
    os.makedirs('models', exist_ok=True)
    model.save('models/cifar10_cnn.h5')
    print("Model saved to models/cifar10_cnn.h5")
