import tensorflow as tf
from tensorflow import keras
import numpy as np

if __name__ == "__main__":
    # Load test data
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test / 255.0
    # Load model
    model = keras.models.load_model('models/cifar10_cnn.h5')
    # Evaluate
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {acc:.4f}")
