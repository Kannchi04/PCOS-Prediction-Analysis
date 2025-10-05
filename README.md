# HerHealth - PCOS Prediction webapp

## Project Overview

This project focuses on building a machine learning model. The workflow includes exploratory data analysis (EDA), data preprocessing, and training several classification models to find the best performer. The final model is saved and deployed in a simple web application.

## Key Files

* `EDA.ipynb`: Contains the initial data exploration, cleaning, and feature engineering.
* `DataAnalysis.ipynb`: Includes data preprocessing, model training, evaluation, and hyperparameter tuning.
* `relevant.csv`: The cleaned dataset with engineered features and the target variable.
* `model.joblib`: The saved machine learning model.
* `scalar.joblib`: The saved `StandardScaler` used for data preprocessing.
* `app.py`: A Python script for the web application to serve predictions.

---

## Exploratory Data Analysis (EDA) & Data Preprocessing

The `EDA.ipynb` notebook handles the initial data preparation steps:

1.  **Loading Data:** The raw dataset is loaded into the notebook.
2.  **Cleaning Data:** Missing values were handled and data types were corrected to prepare the data for analysis.
3.  **Feature Engineering:** Four new, relevant features were created to enhance the model's predictive power.
4.  **Target Variable Creation:** A target variable was defined.

The resulting cleaned dataset, including the new features, was saved as `relevant.csv`.

In the `DataAnalysis.ipynb` notebook, the following steps were taken:

* **Handling Class Imbalance:** The target variable had a significant class imbalance, with the positive class (1) being a minority. To address this, the **Synthetic Minority Over-sampling Technique (SMOTE)** was applied.
* **Feature Scaling:** The features were scaled using `StandardScaler` to ensure all features contribute equally to the model training.
* **Data Splitting:** The dataset was split into training and testing sets to evaluate model performance on unseen data.

---

## Model Training and Evaluation

Several machine learning models were trained and evaluated on the preprocessed data. The performance metrics below were calculated on the **test set** (`X_test`), while the **training set** (`X_train`) metrics were used to check for overfitting.

### Model Performance on `X_test`

| Model               | Accuracy | Precision | Recall   | F1-Score |
|---------------------|----------|-----------|----------|----------|
| Logistic Regression | 87%      | 0.73      | 0.79     | 0.76     |
| Decision Tree       | 87%      | 0.77      | 0.71     | 0.74     |
| Random Forest       | 85%      | 0.70      | 0.75     | 0.72     |
| **XGBoost**         | **92%**  | **0.85**  | **0.82** | **0.84** |
| SVM                 | 88%      | 0.76      | 0.79     | 0.77     |

### Overfitting Check (XGBoost)

To check for overfitting, the XGBoost model's performance on the training data (`X_train`) was evaluated:

* **Accuracy:** 95%
* **Precision:** 0.95
* **Recall:** 0.95
* **F1-Score:** 0.95

The model shows some difference in performance between the training and test sets, indicating a slight degree of overfitting. However, the strong performance on the test set demonstrates that the model generalizes well.

Based on the test set metrics, the **XGBoost model** was chosen as the best-performing model for this project.

---

## Web Application

The best-performing model (`model.joblib`) and the `StandardScaler` (`scalar.joblib`) were saved and integrated into a simple web application using `app.py`. The web app can be used to make new predictions. 

---

## How to Run

1.  Clone this repository: `git clone https://github.com/Kannchi04/PCOS-Prediction-Analysis.git`
2.  Run the web application: `python app.py`

Feel free to explore the notebooks to understand the complete data science pipeline.
