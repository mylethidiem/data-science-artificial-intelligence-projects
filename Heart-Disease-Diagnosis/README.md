# ❤️ Heart Disease Diagnosis — Machine Learning Project

This repository contains a complete end-to-end pipeline for **heart disease prediction** using classical Machine Learning algorithms.
The project explores multiple models, performs feature engineering, evaluates performance, and includes a **Streamlit demo** for basic deployment.

🔗 **Hugging Face Demo:**
[AI VIET NAM AIO2025 MODULE 03 HEART_DISEASE_PREDICTION](https://huggingface.co/spaces/VLAI-AIVN/AIO2025M03_HEART_DISEASE_PREDICTION)

📝 **Notes (Notion):**
[https://www.notion.so/Heart-Disease-Diagnosis-2a40730a967380689b87eeb26a447b72](https://www.notion.so/Heart-Disease-Diagnosis-2a40730a967380689b87eeb26a447b72)

---

## 📌 1. Project Overview

Understanding data is one of the most important steps in machine learning.
In this project, we use the **Cleveland Heart Disease dataset**, a well-known subset from the **UCI Machine Learning Repository**, widely used for benchmarking medical diagnosis models.

This project includes:

* Data preprocessing & cleaning
* Feature engineering
* Model training with raw vs. engineered features
* Classical ML models + ensemble methods
* Performance comparison
* A basic **Streamlit UI** using Decision Tree for deployment

---

## 📂 2. Dataset Description (Cleveland Heart Disease)

The dataset contains **303 patient records** with 14 attributes related to medical examination results.

### 🔎 Feature Explanation

| Feature      | Description                                                                    |
| ------------ | ------------------------------------------------------------------------------ |
| **age**      | Patient age (years)                                                            |
| **sex**      | Gender (1 = male, 0 = female)                                                  |
| **cp**       | Chest-pain type (1 = typical, 2 = atypical, 3 = non-anginal, 4 = asymptomatic) |
| **trestbps** | Resting blood pressure (mmHg)                                                  |
| **chol**     | Serum cholesterol (mg/dl)                                                      |
| **fbs**      | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)                          |
| **restecg**  | Resting ECG results (0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy)     |
| **thalach**  | Maximum heart rate achieved                                                    |
| **exang**    | Exercise-induced angina (1 = yes, 0 = no)                                      |
| **oldpeak**  | ST depression induced by exercise                                              |
| **slope**    | Slope of ST segment (1 = up, 2 = flat, 3 = down)                               |
| **ca**       | Number of major vessels colored by fluoroscopy (0–3)                           |
| **thal**     | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)              |
| **num**      | Target label: 0 = no disease, 1–4 = has disease                                |

---

## ⚙️ 3. Project Pipeline

### **✔ Step 1: Download & Inspect Data**

* Load raw dataset
* Check data types, null values, distributions

### **✔ Step 2: Data Preprocessing**

* Handle missing values
* Convert categorical variables
* Normalize / scale numeric features

### **✔ Step 3: Feature Engineering**

Includes:

* Statistical transformations
* Feature selection
* Domain-specific processing
* Saving cleaned + engineered datasets

### **✔ Step 4: Model Training**

Models trained on **raw** and **feature-engineered** datasets:

#### **Classical ML Models**

* Naive Bayes Classifier
* K-Nearest Neighbors
* Decision Tree

#### **Ensemble Models**

* KNN + Decision Tree + Gaussian Naive Bayes (soft voting)

#### **Boosting / Advanced Models**

* AdaBoost
* Gradient Boosting
* XGBoost
* Random Forest

### **✔ Step 5: Evaluation & Comparison**

* Accuracy, precision, recall, F1
* Confusion matrices
* Model ranking

### **✔ Step 6: Deployment Demo (Streamlit)**

A simple UI using Decision Tree for prediction.

---

## 🗂 4. Repository Structure

```
Heart-Disease-Diagnosis/
│
├── data/
│   ├── raw_and_fe_data/
│   │   ├── raw_train.csv
│   │   ├── raw_val.csv
│   │   ├── raw_test.csv
│   │   ├── fe_train.csv
│   │   ├── fe_val.csv
│   │   ├── fe_test.csv
│   │   ├── fe_feature_names.json
|   |   ├── fe_dt_train.csv
│   │   ├── fe_dt_val.csv
│   │   ├── fe_dt_test.csv
|   |   ├── dt_train.csv
│   │   ├── dt_val.csv
│   │   └── dt_test.csv             
│   └── cleveland.csv
│
├── notebooks/
│   ├── Create_Datasets.ipynb
│   ├── Feature_Engineering.ipynb
│   ├── Decision_Tree.ipynb
│   ├── RandomForest_Diagnosis.ipynb
│   ├── XGBoost_Diagnosis.ipynb
│   ├── KNN_Diagnosis.ipynb
│   ├── Ensemble_Diagnosis.ipynb
│   ├── GradientBoosting_Diagnosis.ipynb
│   └── Deploy_Streamlit.ipynb
│
└── README.md
```

---

## 🛠 5. Technologies Used

* **Python**
* **Pandas**, **NumPy**
* **scikit-learn**
* **XGBoost**
* **Matplotlib / Seaborn**
* **Streamlit**

---

## 🚀 6. How to Run

```bash
# 1. Clone repository
git clone https://github.com/yourname/Heart-Disease-Diagnosis.git
cd Heart-Disease-Diagnosis

# 2. Install requirements
pip install -r requirements.txt

# 3. Run Streamlit app
streamlit run notebooks/Deploy_Streamlit.ipynb
```

---

## 📌 7. Future Improvements

* Add SHAP for model interpretability
* Optimize hyperparameters
* Add FastAPI backend
* Improve UI

---

## ⭐ Acknowledgements

Dataset: **UCI Machine Learning Repository — Cleveland Heart Disease** \
This work is part of my ongoing learning exploration in **Machine Learning and AI**.

