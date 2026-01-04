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
       
     
     
        tenure = int(request.form["tenure"])
        monthly = float(request.form["MonthlyCharges"])
        total = float(request.form["TotalCharges"])
        contract = request.form["Contract"]
        internet = request.form["InternetService"]
        payment = request.form["PaymentMethod"]

    
    
        input_data = {feature: 0 for feature in features}


        input_data["tenure"] = tenure
        input_data["MonthlyCharges"] = monthly
        input_data["TotalCharges"] = total

    
    
        if contract == "One year":
            input_data["Contract_One year"] = 1
        elif contract == "Two year":
            input_data["Contract_Two year"] = 1

       
       
        if internet == "Fiber optic":
            input_data["InternetService_Fiber optic"] = 1

     
     
        payment_map = {
            "Electronic check": "PaymentMethod_Electronic check",
            "Mailed check": "PaymentMethod_Mailed check",
            "Bank transfer (automatic)": "PaymentMethod_Bank transfer (automatic)",
            "Credit card (automatic)": "PaymentMethod_Credit card (automatic)"
        }

        if payment in payment_map:
            input_data[payment_map[payment]] = 1

        input_df = pd.DataFrame([input_data])

        result = model.predict(input_df)[0]
        prediction = "Customer Will Churn" if result == 1 else "Customer Will Not Churn"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

