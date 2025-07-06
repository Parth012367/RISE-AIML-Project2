# 🏦 Loan Eligibility Predictor

A beginner-friendly machine learning project that predicts whether a loan applicant should be approved or not based on their profile using Logistic Regression and Random Forest.

---

## 📌 Problem Statement

Banks want to automate loan approval decisions. This project builds a **classification model** that predicts loan approval based on:

- Age  
- Income  
- Education  
- Credit Score  
- Loan Amount  
and more.

---

## 🎯 Objective

- Preprocess loan applicant data
- Build ML models using:
  - Logistic Regression
  - Random Forest Classifier
- Evaluate using:
  - Confusion Matrix
  - ROC Curve and AUC Score

---

## 🗂️ Project Structure
```
Project 2/
│
├── data/
│ └── loan_data.csv # 📄 Input dataset
│
├── src/
│ ├── preprocess.py # 🔧 Data cleaning and encoding
│ ├── train.py # 🤖 Model training
│ └── evaluate.py # 📊 Model evaluation
│
├── main.py # 🚀 Main runner script
├── requirements.txt # 📦 Python dependencies
└── README.md # 📘 Project instructions
```

---

## 📥 How to Run

### 1. Clone the Repository or Download Project Folder

```bash
git clone https://github.com/yourusername/LoanEligibilityPredictor.git
cd LoanEligibilityPredictor
```
### 2. Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On Linux/macOS
pip install -r requirements.txt
```
### 3. Run the Main Script
```bash 
python main.py
```