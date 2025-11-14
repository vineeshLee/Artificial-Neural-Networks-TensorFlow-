# 🎽 Fashion MNIST Classification with **Keras** & **PyTorch**

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Anaconda](https://img.shields.io/badge/Anaconda-42B029?style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![ML Kit](https://img.shields.io/badge/ML%20Kit-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/ml-kit)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)

---

## 📌 Project Overview

This project demonstrates how to build, train, and evaluate **Convolutional Neural Networks (CNNs)** for classifying Fashion MNIST images using:

- **Keras (TensorFlow backend)**
- **PyTorch**

It also includes converting the trained Keras model into a **TensorFlow Lite (.tflite)** file for deployment on edge and mobile devices.

---

## 👗 Dataset: Fashion MNIST

The **Fashion MNIST** dataset consists of:

- **60,000 training images**
- **10,000 test images**
- **28 × 28 grayscale images**
- **10 clothing categories:**

| Label | Class        |
|-------|--------------|
| 0     | T-shirt/top  |
| 1     | Trouser      |
| 2     | Pullover     |
| 3     | Dress        |
| 4     | Coat         |
| 5     | Sandal       |
| 6     | Shirt        |
| 7     | Sneaker      |
| 8     | Bag          |
| 9     | Ankle Boot   |

---

## 🧠 Models Implemented

### ✅ 1. Keras CNN Model (TensorFlow)

**Architecture:**
- Conv2D → BatchNorm → MaxPool  
- Conv2D → BatchNorm → MaxPool  
- Flatten  
- Dense → Dropout  
- Output Layer  

**Training:**
- **Optimizer:** Adam  
- **Loss:** categorical_crossentropy  
- **Epochs:** 5  
- **Output:** `model.h5`

---

### ✅ 2. PyTorch CNN Model

**Architecture:**
- Conv2d → BatchNorm2d → ReLU → MaxPool2d  
- Conv2d → BatchNorm2d → ReLU → MaxPool2d  
- Flatten  
- Linear → Dropout  
- Output Layer  

**Training:**
- **Optimizer:** Adam  
- **Loss:** CrossEntropyLoss  
- **Epochs:** 5  
- **Output:** `model.pt`

---

## 🔄 TensorFlow Lite Conversion

The trained Keras model is converted to **TensorFlow Lite (.tflite)** for:

- Lightweight mobile deployment  
- Faster inference  
- Smaller storage footprint  
- Use in ML Kit, IoT devices, and embedded systems  

---

## 📊 Results & Evaluation

Both models achieve **high accuracy** on the Fashion MNIST dataset.

The notebook includes:

- Training accuracy & loss curves  
- Validation performance graphs  
- Confusion matrix  
- Prediction samples  

---

## ▶️ How to Run the Project

Run in **Google Colab**, **Jupyter Notebook**, or **local Python environment**.

👤 Author

Created & Maintained by Vineesh Kumar

📧 You can modify the contact section as needed.

📜 License

This project is Completely Free to Use
✔ No restrictions
✔ Open for learning, modification, and distribution
✔ Attribution appreciated but not required

License: Free / Open Use
