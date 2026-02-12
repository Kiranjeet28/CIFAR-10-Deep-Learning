import tensorflow as tf

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs detected: {gpus}")
    else:
        print("No GPU detected. Check your TensorFlow installation and drivers.")

if __name__ == "__main__":
    check_gpu()
