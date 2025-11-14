Fashion MNIST Classification with Keras and PyTorch
This project demonstrates how to build, train, and evaluate Convolutional Neural Networks (CNNs) for image classification on the Fashion MNIST dataset using both Keras (TensorFlow) and PyTorch. Additionally, it includes steps for converting the trained Keras model to a TensorFlow Lite (.tflite) format for deployment on edge devices.

Project Overview
The goal of this project is to classify 10 different categories of clothing items from the Fashion MNIST dataset. We implement two separate CNN models, one using the Keras API (backed by TensorFlow) and another using PyTorch. Both models follow a similar architectural pattern involving convolutional layers, batch normalization, max-pooling, and dense layers.

Dataset
The Fashion MNIST dataset is used, consisting of:

60,000 training images
10,000 test images
Each image is 28x28 grayscale, representing an item of clothing.
There are 10 classes, such as T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.
Models Implemented
1. Keras CNN Model
Architecture: Sequential model with Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, and Dense layers.
Training: Trained for 5 epochs with Adam optimizer and categorical_crossentropy loss.
****: Model saved in .h5 format.
2. PyTorch CNN Model
Architecture: Custom nn.Module with Conv2d, ReLU, BatchNorm2d, MaxPool2d, Flatten, Dropout, and Linear layers.
Training: Trained for 5 epochs with Adam optimizer and CrossEntropyLoss.
Output: Model state dictionary saved in .pt format.
TensorFlow Lite Conversion
The trained Keras model is converted into a TensorFlow Lite (.tflite) format. This format is optimized for mobile and embedded devices, offering reduced model size and lower latency inference.

Results
Both Keras and PyTorch models achieve high accuracy on the Fashion MNIST test set, demonstrating the effectiveness of CNNs for this image classification task. Training and validation loss/accuracy plots are generated for visual analysis of model performance.

How to Run
To reproduce the results, execute the code cells in sequential order within a Google Colab environment or a similar Python environment with TensorFlow and PyTorch installed.

Dependencies
TensorFlow
Keras
PyTorch
torchvision
NumPy
Matplotlib
