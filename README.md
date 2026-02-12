# CIFAR-10 Image Classification with TensorFlow (GPU)
🎯 Project Objective

The objective of this project is to build and improve a deep learning image classification model using the CIFAR-10 dataset.

CIFAR-10 is a benchmark computer vision dataset consisting of 60,000 32x32 color images categorized into 10 distinct classes such as airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

This project aims to:

Understand image data structure and dataset handling

Load and explore the CIFAR-10 dataset using TensorFlow

Develop and experiment with Convolutional Neural Networks (CNNs)

Improve model performance through systematic experimentation

Apply best practices in deep learning project structure

The ultimate goal is to build a scalable, production-ready image classification pipeline while strengthening core deep learning concepts.

## Project Structure

```
project/
│── data/           # Dataset storage
│── notebooks/      # Jupyter notebooks for experiments
│── src/            # Source code (modules, utils)
│── models/         # Saved models
│── logs/           # Training logs
│── train.py        # Training script
│── evaluate.py     # Evaluation script
│── inference.py    # Inference script
│── requirements.txt
│── README.md
│── .gitignore
```

## Setup Instructions

### 1. Create and Activate Virtual Environment


#### Using venv (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Using conda (alternative):
```bash
conda create -n cifar10 python=3.10
conda activate cifar10
```

### 2. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify GPU Availability

Run the following code to check if TensorFlow detects your GPU:

```python
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
# Or run: python src/gpu_check.py
```

### 4. Run Training

```bash
python train.py
```

### 5. Run Evaluation

```bash
python evaluate.py
```

### 6. Run Inference

```bash
python inference.py --image_path path/to/image.png
```


## Best Practices
- Use virtual environments for dependency isolation.
- Keep data, models, and logs in their respective folders.
- Use notebooks for exploration and prototyping.
- Modularize code in `src/` for reusability.
- Track experiments and results in `logs/`.
- Use version control and `.gitignore` to avoid committing large or sensitive files.
- Document your code and process in the README and notebooks.

---

**Happy Training!**
