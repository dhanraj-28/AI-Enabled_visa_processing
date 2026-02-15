# AI Enabled Visa Processing Time Estimator

## 📌 Project Overview
This project predicts visa processing time (in days) using machine learning.

The system uses historical visa application data to estimate how long a visa decision may take based on:
- Processing Center (Employer State)
- Case Status

This project is developed as part of Module 1 of the internship.

---
## Dataset

The dataset used in this project:

Due to GitHub file size limits, the dataset is not included in this repository.


https://www.kaggle.com/datasets/evangelize/h1b-visa-applications

Place the file inside:
data/raw_visa_data.csv









## 🧠 How It Works

1. Raw visa data is cleaned and preprocessed.
2. Processing time (in days) is calculated from:
   - Application Date
   - Decision Date
3. Categorical features are encoded using Label Encoding.
4. A machine learning regression model is trained.
5. A Streamlit frontend allows users to:
   - Select Processing Center
   - Select Case Status
   - Get predicted processing time

---

## 📂 Project Structure

internship/
│
├── data/
│ ├── raw_visa_data.csv
│ └── cleaned_data.csv
│
├── models/
│ ├── visa_model.pkl
│ ├── center_encoder.pkl
│ └── status_encoder.pkl
│
├── src/
│ ├── preprocessing.py
│ └── train_model.py
│
├── frontend/
│ └── app.py
│
└── README.md


---

## ⚙️ Installation

Create virtual environment:



python -m venv venv


Activate environment:

**Windows**


venv\Scripts\activate
