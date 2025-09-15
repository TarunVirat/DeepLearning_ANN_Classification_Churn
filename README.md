# 📊 Deep Learning ANN Classification - Customer Churn Prediction

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen)](https://deeplearningannclassificationchurn-e3vacbrkaf2g4pwudrsswn.streamlit.app/)

This project implements an **Artificial Neural Network (ANN)** to predict **customer churn** in the banking sector.  
The deployed web app is built using **Streamlit** and allows users to input customer details and receive a **churn probability prediction** in real-time.

---

## 🚀 Features
- Interactive **Streamlit app** (`app.py`) for churn prediction  
- Pre-trained **ANN model** (`model.h5`) built with **TensorFlow/Keras**  
- Encoders and scaler (`label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, `scaler.pkl`) for consistent preprocessing  
- Clean UI for entering customer details (credit score, age, balance, geography, etc.)  
- Outputs both **churn probability** and a **decision (likely to churn / not likely)**  

---

## 🛠️ Tech Stack
- **Python**
- **TensorFlow / Keras**
- **scikit-learn / SciKeras**
- **Pandas / NumPy**
- **Matplotlib**
- **Streamlit**

---

## 📂 Project Structure
├── app.py # Streamlit app (user interface + prediction pipeline)
├── requirements.txt # Python dependencies
├── Churn_Modelling.csv # Dataset (10k customers)
├── model.h5 # Trained ANN model
├── scaler.pkl # StandardScaler for numeric features
├── onehot_encoder_geo.pkl # OneHotEncoder for 'Geography'
├── label_encoder_gender.pkl # LabelEncoder for 'Gender'
├── prediction.ipynb # Notebook for predictions
├── experiments.ipynb # Model experiments
├── hyperparametertuningann.ipynb # Hyperparameter tuning
├── salaryregression.ipynb # Regression example

---

## 📊 Dataset
The dataset (`Churn_Modelling.csv`) contains information about 10,000 bank customers, including:
- **Demographics**: Age, Gender, Geography  
- **Financial details**: Balance, Credit Score, Estimated Salary  
- **Banking behavior**: Tenure, Number of Products, Has Credit Card, Is Active Member  
- **Target variable**: `Exited` (1 = churn, 0 = retained)  

---

## 🔮 Model Workflow
1. **Preprocessing**:
   - Label Encoding → Gender  
   - One-Hot Encoding → Geography  
   - Standard Scaling → Numerical features  
2. **Neural Network**:
   - Input layer → Hidden layers → Output sigmoid layer  
   - Trained with **Binary Crossentropy Loss** & **Adam Optimizer**  
3. **Prediction**:
   - Returns churn probability (0–1)  
   - Threshold at 0.5 → Likely to churn / Not likely  

---

## ▶️ Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/deeplearning_ann_classification_churn.git
   cd deeplearning_ann_classification_churn
