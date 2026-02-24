# Customer Churn Prediction System

##  Project Overview
Customer churn is a critical business problem in the telecom industry. Acquiring new customers is significantly more expensive than retaining existing ones.  
This project builds an **end-to-end machine learning system** to predict whether a customer is likely to churn based on their service usage, contract details, and billing information.

The project covers the **entire data science lifecycle**:
- Data preprocessing
- Exploratory data analysis (EDA)
- Feature engineering
- Model training and evaluation
- Model deployment using Flask

---

##  Objective
To predict customer churn (`Yes / No`) and identify key drivers contributing to churn, enabling proactive retention strategies.

---

##  Tech Stack
- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Web Framework:** Flask
- **Model Serialization:** Joblib

## Project Structure
<pre markdown="1">
customer-churn-prediction/
├── params.yaml
├── README.md
├── requirements.txt
├── app/
│ ├── app.py
│ └── templates/
│ └── index.html
├── config/
│ └── config.yaml
├── data/
│ ├── raw/
│ │ └── telco_churn.csv
│ └── processed/
│ ├── churn_cleaned.csv
│ └── churn_final_eda.csv
├── image/
│ ├── churn_distribution.png
│ ├── correlation_heatmap.png
│ ├── feature_correlation_with_churn.png
│ ├── MonthlyCharges_distribution.png
│ ├── monthlycharges_vs_churn.png
│ ├── pairplot.png
│ ├── tenure_distribution.png
│ ├── tenure_vs_churn.png
│ ├── TotalCharges_distribution.png
│ └── totalcharges_vs_churn.png
├── model/
│ ├── churn_model.pkl
│ └── features.pkl
├── notebooks/
│ └── 01_eda.ipynb
└── src/
├── 01_data_preprocessing.py
├── 02_train_model.py
└── 03_evaluate_model.py</pre>

##  Project Overview
Customer churn is a critical business problem in the telecom industry. Acquiring new customers is significantly more expensive than retaining existing ones.  
This project builds an **end-to-end machine learning system** to predict whether a customer is likely to churn based on their service usage, contract details, and billing information.

The project covers the **entire data science lifecycle**:
- Data preprocessing
- Exploratory data analysis (EDA)
- Feature engineering
- Model training and evaluation
- Model deployment using Flask

---

##  Objective
To predict customer churn (`Yes / No`) and identify key drivers contributing to churn, enabling proactive retention strategies.

---

##  Tech Stack
- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Web Framework:** Flask
- **Model Serialization:** Joblib


## Dataset
Name: Telco Customer Churn (IBM Sample Dataset)

Source: Kaggle
 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

 ## Model Training
**Algorithm**: Random Forest Classifier

**Class Imbalance Handling**: class_weight="balanced"

**Train-Test Split**: Stratified split to preserve churn distribution

**Artifacts Saved**:

    Trained model (churn_model.pkl)

    Feature names (features.pkl) to ensure consistent inference

**Model Performance**

    Achieved approximately 80% accuracy

    Demonstrated strong performance in identifying high-risk churn customers


## Web Application (Flask)

A Flask-based web application was developed to deploy the trained model.

**Features**

    Accepts customer input via a web form

    Performs real-time churn prediction

    Displays prediction results instantly

## How to Run the Project
**1. Clone the Repository**

    git clone https://github.com/Pranish-Sapkota/customer-churn-prediction

    cd customer-churn-prediction


**2. Create a Virtual Environment**

    python -m venv venv

    source venv/bin/activate 


**3. Install Dependencies**

    pip install -r requirements.txt


**4. Train the Model**

    python src/01_data_preprocessing.py

    python src/02_train_model.py

    python src/03_evaluate_model.py



**5. Run the Flask App**

    python app/app.py





