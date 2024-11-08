# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import pickle

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

# Function to train the Decision Tree model
def train_model(X_train, y_train, max_depth=10, min_samples_split=5, random_state=42):
    regressor = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
    regressor.fit(X_train, y_train)
    return regressor

# Function to evaluate the model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_score = model.score(X_train, y_train) * 100
    test_score = model.score(X_test, y_test) * 100
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Training Score: {train_score:.2f}%")
    print(f"Test Score: {test_score:.2f}%")
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    
    return mse, rmse

# Function to make predictions with a given model
def make_predictions(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

# Function to evaluate the predictions
def evaluate_predictions(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    return mse, rmse

# Function to perform hyperparameter tuning using GridSearchCV
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5]
    }
    grid_search = GridSearchCV(
        DecisionTreeRegressor(random_state=42), 
        param_grid, 
        cv=5, 
        scoring="neg_mean_squared_error"
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Main function to run the entire pipeline
def main():
    # Load and inspect data
    df = load_and_inspect_data(r"data\cleaned\properties.csv")
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    mse, rmse = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Make and evaluate predictions
    y_pred = make_predictions(model, X_test)
    mse, rmse = evaluate_predictions(y_test, y_pred)
    
    # Perform hyperparameter tuning
    best_model = hyperparameter_tuning(X_train, y_train)
    
    # Evaluate the best model
    y_pred_best = make_predictions(best_model, X_test)
    mse_best, rmse_best = evaluate_predictions(y_test, y_pred_best)
    
    print("Best Model Mean Squared Error:", mse_best)
    print("Best Model Root Mean Squared Error:", rmse_best)


# Run the main function
if __name__ == "__main__":
    main()
