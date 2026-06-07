# 🤟 ASL Recognition System

A real-time American Sign Language (ASL) recognition system that uses MediaPipe hand landmarks and a Machine Learning (MLP) model to convert hand gestures into text and speech.

## 🚀 Features

* Real-time ASL gesture recognition
* MediaPipe hand landmark detection (21 landmarks)
* MLP-based classification model
* Text generation from hand gestures
* Text-to-Speech (TTS) support
* Streamlit-based interactive web interface
* Support for A-Z alphabets, Space, and Delete gestures

## 🛠️ Technologies Used

* Python
* MediaPipe
* OpenCV
* Scikit-Learn
* Streamlit
* Streamlit-WebRTC
* Pyttsx3

## 📊 Model Information

* Dataset: ASL Alphabet Dataset (Kaggle)
* Feature Extraction: 21 Hand Landmarks (63 Features)
* Model: Multi-Layer Perceptron (MLP)
* Accuracy: 99.05%

## 🔄 Workflow

```text
Webcam
   ↓
MediaPipe Hands
   ↓
21 Hand Landmarks
   ↓
Feature Vector (63 Features)
   ↓
MLP Classifier
   ↓
Letter Prediction
   ↓
Word Formation
   ↓
Text-To-Speech
```

## 📂 Project Structure

```text
ASL_Project/
│
├── models/
│   ├── asl_model.pkl
│   └── label_encoder.pkl
│
├── app.py
├── webcam_predict.py
├── requirements.txt
└── README.md
```

## ▶️ Installation

```bash
pip install -r requirements.txt
```

## ▶️ Run Locally

```bash
streamlit run app.py
```

## 🎯 Future Enhancements

* Support for complete words and sentences
* Mobile-friendly deployment
* Multi-hand gesture recognition
* Enhanced UI/UX
* Cloud deployment with Streamlit Cloud

## 👨‍💻 Author
1) Priyansu pal
2) Swati Rani Bhanja
3)Himani Dash
4)Somesh Ranjan Nayak
