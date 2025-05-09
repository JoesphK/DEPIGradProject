# Diabetes Prediction App
A machine learning web application using the DiaHealth Bangladeshi dataset to predict the risk of Type 2 Diabetes. 

This app is built using Python, trained in a Jupyter Notebook, deployed and served with a Streamlit UI.

## 📊 Dataset

- **Source:** [Mendeley Data – DiaHealth](https://data.mendeley.com/datasets/7m7555vgrn/1)
- **Title:** *DiaHealth: A Bangladeshi Dataset for Type 2 Diabetes Prediction*
- **Size:** 5,437 rows × 15 columns
- **License:** Open for research purposes

The dataset contains medical information relevant to diabetes prediction.

## 📑 Dataset Metadata

This dataset contains **5,437 records** and **15 features** (medical and demographic). Below is a description of each column:

| #   | Column Name              | Type     | Description                              |
|-----|--------------------------|----------|------------------------------------------|
| 0   | `age`                    | int64    | Age of the patient                       |
| 1   | `gender`                 | object   | Gender (`Male`, `Female`, etc.)          |
| 2   | `pulse_rate`            | int64    | Pulse rate (beats per minute)            |
| 3   | `systolic_bp`           | int64    | Systolic blood pressure                   |
| 4   | `diastolic_bp`          | int64    | Diastolic blood pressure                  |
| 5   | `glucose`               | float64  | Blood glucose level (mmol/L)              |
| 6   | `height`                | float64  | Height in meters                          |
| 7   | `weight`                | float64  | Weight in kilograms                       |
| 8   | `bmi`                   | float64  | Body Mass Index                           |
| 9   | `family_diabetes`       | int64    | Family history of diabetes (1 = Yes)     |
| 10  | `hypertensive`          | int64    | Hypertension status (1 = Yes)            |
| 11  | `family_hypertension`   | int64    | Family history of hypertension            |
| 12  | `cardiovascular_disease`| int64    | Presence of cardiovascular disease        |
| 13  | `stroke`                | int64    | History of stroke                        |
| 14  | `diabetic`              | object   | Target label (`Yes` or `No`)              |

## 📌 Feature Selection & Class Balancing

### ⚖️ 1. Addressing Class Imbalance with SMOTE

- The dataset was **imbalanced**, with more non-diabetic cases than diabetic ones.
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic examples for the minority class (diabetic cases).
- SMOTE was applied **before** feature selection to prevent data leakage and ensure that both synthetic and real instances were used for feature selection.
  
> SMOTE helps improve model recall for the minority class by balancing the data distribution.

### 🧮 2. Feature Selection with ANOVA

- After addressing class imbalance, **ANOVA F-test** (`f_classif`) was applied to assess the relationship between each feature and the target variable (`diabetic`).
- Features with significant scores (p < 0.05) were retained for model training.

**Final Selected Features**:

After applying SMOTE, the following **features** were selected based on their significance with the target (`diabetic`):

| Feature        | Type     | Log Transformed |
|----------------|----------|-----------------|
| `glucose`      | Numeric  | ✅               |
| `hypertensive` | Boolean  | ❌               |
| `diastolic_bp` | Numeric  | ❌               |
| `systolic_bp`  | Numeric  | ❌               |
| `weight`       | Numeric  | ✅               |
| `bmi`          | Numeric  | ✅               |
| `age`          | Numeric  | ❌               |

✅ = Feature was **log-transformed** using `np.log1p()` to reduce skewness.

---

> These selected features were then scaled and used for model training in the Streamlit app.



## 🧹 Data Preprocessing

Before training the model, the following preprocessing steps were applied:

- Converted categorical columns like `hypertensive` and `diabetic` into numerical values
- Handled any missing values and duplicates
- Split the dataset into training and test sets (e.g., 80/20 split)

## 🤖 Model Training & Evaluation

After preprocessing and feature selection, we applied two machine learning models to predict **Type 2 Diabetes**:

### 1. **XGBoost**

- **XGBoost** is an optimized gradient boosting library that performs well on structured/tabular data.

### 2. **Random Forest**

- **Random Forest** is an ensemble learning method that combines multiple decision trees to improve predictive accuracy and control overfitting.

### 📝 Model Evaluation

Both models were evaluated using the following metrics:

- **Accuracy**: Proportion of correct predictions
- **Precision**: The percentage of positive predictions that were correct
- **Recall**: The percentage of actual positives that were correctly identified
- **F1-score**: The weighted average of Precision and Recall

### 📊 Classification Reports

**XGBoost Classification Report:**
```plaintext
        class   precision   recall  f1-score   

        0.0     0.94        0.89    0.92       
        1.0     0.90        0.95    0.92       

    accuracy                        0.92       
```

**Random Forest Classification Report:**
```plaintext
        class   precision   recall  f1-score   

        0.0     0.94        0.91    0.92       
        1.0     0.91        0.95    0.93      

    accuracy                        0.93       
```

### 💡 Conclusion

- **XGBoost** achieved an **accuracy** of **92%**, with a precision of 0.94 and recall of 0.89 for the negative class (0.0) and precision of 0.90 and recall of 0.95 for the positive class (1.0).
- **Random Forest** performed slightly better, achieving an **accuracy** of **93%**, with a precision of 0.94 and recall of 0.91 for the negative class (0.0) and precision of 0.91 and recall of 0.95 for the positive class (1.0).
- Both models demonstrate high performance in predicting Type 2 Diabetes, with **Random Forest** slightly outperforming **XGBoost** in terms of accuracy.

### 🚀 Deployment

To deploy this diabetes prediction app, you can follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/JoesphK/DEPIGradProject.git
    cd diabetes-prediction-app
    ```

2. **Install dependencies**:
    Ensure that you have `Python 3.x` and `pip` installed. Then install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:
    For deploying the app locally, we are using **Streamlit**. To start the app, run the following command:
    ```bash
    streamlit run app.py
    ```

4. **Interact with the app**:
    Once the app is running, you can access it locally at `http://localhost:8501`. The app will prompt you to enter values such as age, glucose level, blood pressure, etc., and predict the likelihood of Type 2 Diabetes.

### 🌐 Streamlit Deployment

Streamlit also allows you to deploy the application to the web, making it accessible to a wider audience. You can use platforms like **Streamlit Sharing** or **Heroku** for easy deployment without needing extensive setup.

To deploy on **Streamlit Sharing**, follow these steps:

1. Push your code to a GitHub repository or fork the remote repository.
2. Create an account on [Streamlit Sharing](https://streamlit.io/sharing).
3. Link your GitHub repository to Streamlit Sharing, and it will automatically deploy the app.
4. Share the generated link with others to access the app on the web.
