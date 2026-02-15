import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train():

    df = pd.read_csv("data/cleaned_data.csv")

    # Features
    X = df[['processing_center', 'case_status']]

    # Target
    y = df['processing_days']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor()

    print("Training started...")
    model.fit(X_train, y_train)
    print("Training completed!")

    # Save model
    joblib.dump(model, "models/visa_model.pkl")
    print("Model saved successfully!")

if __name__ == "__main__":
    train()
