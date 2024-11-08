import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Function to load the trained model using joblib
def load_model(file_path):
    try:
        model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model
    except FileNotFoundError:
        print(f"Model file not found at {file_path}. Training a new model.")
        return None

# Function to load and inspect data
def load_and_inspect_data(file_path):
    df = pd.read_csv(file_path)
    print(df.head())
    print(df.isnull().sum())
    print("Total missing values:", df.isnull().sum().sum())
    df.info()
    return df

# Function to preprocess data
def preprocess_data(df):
    # Separate features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Handle missing values
    X.fillna("MISSING", inplace=True)
    y.fillna(y.mean(), inplace=True)

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Standardize numeric features
    scaler = StandardScaler()
    numeric_columns = X.select_dtypes(include=["float64", "int64"]).columns
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    return X, y

# Function to split data into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Function to make predictions and calculate the score
def predict_and_score(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate squared predictions
    y_pred_squared = y_pred ** 2
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Calculate MSE of squared predictions
    mse_squared = mean_squared_error(y_test ** 2, y_pred_squared)
    
    return y_pred, y_pred_squared, mse, rmse, mse_squared

# Function to train and save the model
def train_and_save_model(X_train, y_train, model_path="model_filename.pkl"):
    # Ensure the directory exists
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # Create the directory if it doesn't exist
    
    # Initialize and train the model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)


    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model saved as {model_path}")
    return model

# Define the main function to run the entire pipeline
def main():
    # Load and inspect data
    df = load_and_inspect_data(r"data\cleaned\properties.csv")  # Adjust the file path
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Load or train the model
    model_path = r"C:\Users\yusra\OneDrive\Documents\Models\model_filename.pkl"  # Adjust the model path
    model = load_model(model_path)  # Try loading the saved model
    
    if model is None:  # If the model was not found, train a new one
        model = train_and_save_model(X_train, y_train, model_path)
    
    # Make and evaluate predictions
    y_pred, y_pred_squared, mse, rmse, mse_squared = predict_and_score(model, X_test, y_test)
    
    # Print evaluation metrics
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("MSE of Squared Predictions:", mse_squared)



# Run the main function
if __name__ == "__main__":
    main()
