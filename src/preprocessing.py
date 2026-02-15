import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data(input_path, output_path):

    df = pd.read_csv(input_path, low_memory=False)

    print("\nColumns found in dataset:")
    print(df.columns)

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Rename columns based on YOUR dataset
    df.rename(columns={
        'case_submitted': 'application_date',
        'decision_date': 'decision_date',
        'employer_state': 'processing_center'
    }, inplace=True)

    # Check required columns
    required_columns = ['application_date', 'decision_date', 'processing_center', 'case_status']

    for col in required_columns:
        if col not in df.columns:
            print(f"ERROR: Required column '{col}' not found.")
            return

    # Convert dates
    df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce')
    df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')

    # Remove rows with missing dates
    df.dropna(subset=['application_date', 'decision_date'], inplace=True)

    # Create target variable
    df['processing_days'] = (
        df['decision_date'] - df['application_date']
    ).dt.days

    # Remove negative processing time
    df = df[df['processing_days'] >= 0]

    # Fill missing values
    df['processing_center'] = df['processing_center'].fillna('Unknown')
    df['case_status'] = df['case_status'].fillna('Unknown')

    # -------------------------
    # Encode categorical columns
    # -------------------------
    le_center = LabelEncoder()
    le_status = LabelEncoder()

    df['processing_center'] = le_center.fit_transform(df['processing_center'])
    df['case_status'] = le_status.fit_transform(df['case_status'])

    # -------------------------
    # Save encoders (VERY IMPORTANT)
    # -------------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(le_center, "models/center_encoder.pkl")
    joblib.dump(le_status, "models/status_encoder.pkl")

    print("✅ Encoders saved successfully!")

    # Final features
    final_df = df[['processing_center', 'case_status', 'processing_days']]

    # Save cleaned dataset
    final_df.to_csv(output_path, index=False)

    print("\n✅ Module 1 preprocessing completed successfully!")
    print(f"Cleaned dataset saved at: {output_path}")


if __name__ == "__main__":
    preprocess_data(
        "data/raw_visa_data.csv",
        "data/cleaned_data.csv"
    )
