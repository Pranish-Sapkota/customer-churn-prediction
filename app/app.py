from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "model", "churn_model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "model", "features.pkl"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # RAW USER INPUT
        tenure = int(request.form["tenure"])
        monthly = float(request.form["MonthlyCharges"])
        total = float(request.form["TotalCharges"])
        contract = request.form["Contract"]
        internet = request.form["InternetService"]
        payment = request.form["PaymentMethod"]

        # Initialize all features to 0
        input_data = dict.fromkeys(features, 0)

        # Numeric values
        input_data["tenure"] = tenure
        input_data["MonthlyCharges"] = monthly
        input_data["TotalCharges"] = total

        # One-hot encoding (MATCH TRAINING)
        if contract != "Month-to-month":
            input_data[f"Contract_{contract}"] = 1

        if internet != "No":
            input_data[f"InternetService_{internet}"] = 1

        input_data[f"PaymentMethod_{payment}"] = 1

        input_df = pd.DataFrame([input_data])

        result = model.predict(input_df)[0]

        prediction = "Customer Will Churn" if result == 1 else "Customer Will Not Churn"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
