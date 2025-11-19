# Customer Churn Prediction

**Objective:**  
Identify bank customers who are likely to leave (churn) using machine learning techniques.

This project uses the **Churn Modelling Dataset** and demonstrates data cleaning, encoding, model training, and feature interpretation.

## 📌 Overview

The goal of this project is to build a predictive model that classifies whether a customer will leave the bank (`Exited = 1`) or stay (`Exited = 0`).  

The analysis includes:

- Cleaning and preparing the dataset  
- Encoding categorical features (Gender, Geography)  
- Training a classification model (Random Forest)  
- Analyzing feature importance to understand churn factors  

## 📂 Files in This Repository

- **`customer_churn_prediction.py`** — Main script for data cleaning, modeling, and evaluation  
- **`Churn_Modelling.csv`** — Dataset used for customer churn analysis
- **`feature_importance.png`** — Feature importance visualization (uploaded directly)
- **`README.md`** — Project documentation

## 📘 Steps Performed in the Project

### **1. Data Loading & Cleaning**
- Removed irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
- Checked for missing values
- Ensured dataset consistency

### **2. Encoding Categorical Variables**
- **Gender** → Label Encoding  
- **Geography** → One-Hot Encoding  

### **3. Train-Test Split & Scaling**
- Standard scaling applied to numerical features  

### **4. Model Training**
A **Random Forest Classifier** is used because:
- It handles mixed data types  
- Offers strong performance  
- Provides feature importance  

### **5. Evaluation Metrics**
- Accuracy score  
- Confusion matrix  
- Classification report  

### **6. Feature Importance**
A bar plot shows the influence of:
- Credit Score  
- Age  
- Balance  
- Estimated Salary  
- Geography categories  

## ▶️ How to Run the Project

1. Install the required libraries:
pip install pandas numpy seaborn matplotlib scikit-learn
2. Download the dataset (Churn_Modelling.csv) and place it in the project folder.
3. Run the Python script.
