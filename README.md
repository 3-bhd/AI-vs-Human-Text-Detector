# AI vs Human Text Detection — Phase V  
Real-Time Deployment • Flask API • Web Interface • Automated Retraining

This repository contains the full deployment of an AI–vs–Human text classifier using a hybrid TF–IDF and stylistic feature representation with a Linear SVM model. The system includes a Flask backend, a browser-based interface, a CLI client, and an automated offline retraining pipeline that redeploys through GitHub → Railway CI/CD.

The deployed application is available at:  
https://web-production-efc42.up.railway.app/

---

## Features
- Real-time classification via a Flask REST API  
- Web-based GUI (HTML/JS) for predictions and feedback  
- Command-line client for local testing  
- Feedback logging through `/feedback`  
- Automated offline retraining using accumulated feedback  
- Automatic redeployment through CI/CD  
- Modular preprocessing and inference pipeline

---

## Installation
```bash
pip install -r requirements.txt
Python 3.9+ is required.
```
Running the Backend
```bash
python api_server.py
```
Local endpoints:

GET / — Web interface

POST /predict — Run inference

POST /feedback — Store corrected label

GET /download-feedback?key=1234 — Export feedback

GET /health — Service status

Command-Line Client
```bash
python cli_client.py
```
Sends input text to the backend, displays predictions, and optionally submits feedback.

Web Interface
Access locally at:

http://localhost:8000

Deployed version:

https://web-production-efc42.up.railway.app/

Supports prediction, confidence scores, feedback submission, and admin feedback export.

## Repository Structure

├── api_server.py              # Flask backend (prediction, feedback, admin tools)

├── cli_client.py              # Command-line client for local testing

├── inference_svm.py           # Vectorization, feature scaling, SVM inference

├── preprocess_utils.py        # Text cleaning + numeric stylistic features

├── train_linear_svm.py        # Final model training script

├── retrain_with_feedback.py   # Automatic offline retraining + CI/CD integration

├── artifacts/                 # Serialized model, vectorizer, scaler

│   └── models/

├── static/                    # Web-based GUI (index.html + assets)

│   └── index.html

├── requirements.txt           # Python dependencies

├── Procfile                   # Deployment configuration for Railway

└── README.md


## Authors
Ahmed Monir

Omar Abdelhady

The American University in Cairo
