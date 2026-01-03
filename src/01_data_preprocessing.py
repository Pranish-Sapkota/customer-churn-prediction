import pandas as pd

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Drop ID
    df.drop("customerID", axis=1, inplace=True)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Binary columns
    binary_cols = [
        "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "Churn"
    ]

    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # One-hot encode categorical variables
    categorical_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df.to_csv(output_path, index=False)
    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_data(
        "data/raw/telco_churn.csv",
        "data/processed/churn_cleaned.csv"
    )
