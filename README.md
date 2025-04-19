# 🛡️ Insurance Fraud Detection using Logistic Regression

This project implements a logistic regression model to classify insurance claims as **fraudulent (1)** or **non-fraudulent (0)**. It demonstrates a complete machine learning pipeline including data preprocessing, class imbalance handling, model training, evaluation, and insights.

---

## 📊 Dataset

The dataset `insurance_claims.csv` contains:
- Policyholder information
- Vehicle and claim details
- A target label `Fraud` (1 = Fraudulent, 0 = Legitimate)

---

## 🔧 Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Statistical summary of dataset
- Class distribution visualization

### 2. Preprocessing
- Dropped irrelevant ID and text columns
- Label-encoded categorical variables
- Scaled numerical features using `StandardScaler`

### 3. Handling Class Imbalance
- Used **SMOTE (Synthetic Minority Oversampling Technique)** to oversample the minority (fraud) class

### 4. Model Training
- Trained a **Logistic Regression** model with:
  - `class_weight='balanced'`
  - `solver='saga'`
  - `max_iter=1000`

### 5. Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **AUC-ROC**
- Confusion Matrix visualization


