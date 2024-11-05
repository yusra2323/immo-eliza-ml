
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from preprocessing import DataPreprocessor
import joblib
from xgboost import XGBRegressor
import argparse

# Load the dataset
def load_data():
    data = pd.read_csv(r"data\raw\properties.csv")
    
    # Separate features (X) and target variable (y)
    X = data.drop(columns=['price'])  # Replace 'price' with your target column name
    y = data['price']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Preprocess the data
def preprocess_data(X_train, X_test):

    # Define numeric and categorical columns
    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # Define preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform the training data, and transform the test data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    # Save the preprocessor for use in predict.py
    joblib.dump(preprocessor, 'preprocessor.pkl')
    
    return X_train, X_test

# Train the XGBoost model
def train_model(X_train, y_train):
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    return model

# Main function to load data, preprocess, train, and evaluate the model

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_data(r"data\cleaned\properties.csv")
    X_train, X_test = preprocess_data(X_train, X_test)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, 'trained_xgb_model.pkl')
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Model RMSE on test set: {rmse:.2f}")

# Execute the main function when the script is run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model for property price prediction")
    parser.add_argument("filepath", type=str, help="Path to the CSV file containing the dataset")
    args = parser.parse_args()
    
    main(args.filepath)


