import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import argparse

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main(image_path):
    model = keras.models.load_model('models/cifar10_cnn.h5')
    img = preprocess_image(image_path)
    preds = model.predict(img)
    class_idx = np.argmax(preds, axis=1)[0]
    print(f"Predicted class: {CLASSES[class_idx]} ({class_idx})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR-10 Inference')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    args = parser.parse_args()
    main(args.image_path)
