#  Metallic Surface Defect Detection using CNN + ViT

##  Project Overview

This project detects defects on metallic surfaces using a hybrid deep learning model combining Convolutional Neural Networks (CNN) and Vision Transformer (ViT).

##  Model Architecture

* CNN for feature extraction
* Vision Transformer for global pattern learning
* Two-phase training for improved accuracy

##  Features

* Detects 6 types of defects:

  * Crazing, Inclusion, Patches, Pitted, Rolled, Scratches
* Flask API for backend prediction
* React frontend for user interaction
* Grad-CAM visualization for explainability

##  Technologies Used

* Python, PyTorch
* Flask (Backend API)
* React + Vite (Frontend)
* NumPy, OpenCV, Matplotlib

##  Project Structure

* Backend: Flask API (app.py)
* Model: CNN + ViT (model.py)
* Training: train.py
* Dataset Preparation: prepare_dataset.py
* Frontend: React

##  How to Run

### Backend:

pip install -r requirements.txt
python app.py

### Frontend:

npm install
npm run dev

##  Dataset

Dataset used: NEU Metal Surface Defects Dataset


##  Results

* Achieves high accuracy using two-phase training
* Improved generalization using ViT

##  Author
Maalees Sushma

Your Name
