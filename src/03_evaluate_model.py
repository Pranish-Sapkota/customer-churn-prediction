import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/processed/churn_final_eda.csv")

X = df.drop('Churn', axis=1)
y = df['Churn']

model = joblib.load("model/churn_model.pkl")

y_pred = model.predict(X)

print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
