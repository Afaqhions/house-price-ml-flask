import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
import joblib
import os

def train_and_save_model():
    print("Fetching data...")
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Price'] = data.target

    # Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
    X = df.drop('Price', axis=1)
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    joblib.dump(model, 'models/house_price_model.pkl')
    joblib.dump(data.feature_names, 'models/features.pkl')
    print("Model saved to models/house_price_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
