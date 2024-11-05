import pandas as pd
import joblib
from preprocessing import DataPreprocessor



  #Load the trained RandomForest model from the specified path."""
def load_model(model_path):
  
    return joblib.load(model_path)


def preprocess_new_data(data_path):
   
    data_preprocessor = DataPreprocessor(data_path)
    preprocessed_data = data_preprocessor.preprocess()
    return preprocessed_data

 #Make predictions using the loaded model and preprocessed new data.
def make_predictions(model, new_data):
   
    predictions = model.predict(new_data)
    return predictions

# Save the predictions to a CSV file.
def save_predictions(predictions, output_path):
   
    pd.DataFrame(predictions, columns=['Predicted Price']).to_csv(
        output_path, index=False)
    print(f"Predictions saved to {output_path}")


 # Path to the Cleaned_2.csv data (to be predicted)
if __name__ == "__main__":
   
    new_data_path = 'data/Cleaned.csv'


    # Path to the saved RandomForest model
    model_path = 'random_forest_model.joblib'

    # Load the trained RandomForest model
    model = load_model(model_path)

    # Preprocess the new data
    new_data = preprocess_new_data(new_data_path).drop(
        'price', axis=1, errors='ignore')

    # Make predictions on the new data
    predictions = make_predictions(model, new_data)

    # Save or display the predictions
    save_predictions(predictions, 'predictions_on_Cleaned.csv')