## Hand Gesture Recognition using MobileNetV2

This project implements a **real-time hand gesture recognition system** using **TensorFlow**, **MobileNetV2**, **MediaPipe**, and **OpenCV**.  
It trains a deep learning model on an image dataset of hand gestures and then uses a webcam to recognize gestures live.

---

## Project Overview

The project consists of **three major stages**:

1. Training a CNN model using transfer learning (**MobileNetV2**)
2. Loading the trained model
3. Real-time hand gesture prediction using a webcam

**MediaPipe** is used for hand detection, while the trained CNN performs gesture classification.

## Model Architecture

The model uses **MobileNetV2** as a feature extractor with custom classification layers.

### 🔹 Base Model
- **MobileNetV2** (pretrained on ImageNet)
- Input Shape: `(128, 128, 3)`
- Pretrained layers are **frozen**

### 🔹 Custom Classification Layers
- Global Average Pooling
- Dense layer (128 units, ReLU)
- Dropout (0.4)
- Output Dense layer with Softmax activation

This approach improves accuracy while keeping the model lightweight and fast.

---
## Technologies Used

- **TensorFlow / Keras** – Deep learning
- **MobileNetV2** – Transfer learning model
- **MediaPipe** – Hand detection & landmarks
- **OpenCV** – Image processing & webcam handling
- **NumPy** – Numerical operations

---
## Output:
  -**live_webcam_feed**: Displays real-time video from the webcam
  
  -**hand_bounding_box**: Shows bounding box around the detected hand
  
  -**predicted_gesture_name**: Displays the recognized hand gesture
  
  -**confidence_percentage**: Shows model prediction confidence
  
  -**hand_landmark_visualization**: Draws hand landmarks using MediaPipe
