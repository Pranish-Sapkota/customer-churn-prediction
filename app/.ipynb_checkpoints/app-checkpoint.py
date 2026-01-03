from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("../model/churn_model.pkl")
features = joblib.load("../model/features.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        data = {}
        for feature in features:
            data[feature] = float(request.form.get(feature, 0))

        df = pd.DataFrame([data])
        result = model.predict(df)[0]

        prediction = "Customer Will Churn" if result == 1 else "Customer Will Not Churn"

    return render_template("index.html", prediction=prediction, features=features)

if __name__ == "__main__":
    app.run(debug=True)
