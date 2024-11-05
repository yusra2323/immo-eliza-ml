# immo-eliza-ml
# ImmoWeb Real Estate machine learning



## Project Overview
This project aims to build a machine learning model that predicts property prices in Belgium using data from Immoweb, Belgium's leading real estate website. The primary goal is to assist Immo Eliza with a reliable tool for estimating property prices based on various features like location, size, and other property characteristics. This README provides details on setting up, processing the data, training models, and using the FastAPI service to access predictions.

## Dataset
The dataset is built by scraping data from Immoweb and includes over 10,000 properties with the following attributes:


Dataset


	1.	Source: The dataset can be scraped from ImmoWeb .

	2.	Features: Typical features include:

	•	Location: Property location (city, regin).

	•	Size: Size of the property in square meters.

	•	Bedrooms: Number of bedrooms.

	•	Property Type: House, apartment, etc.

	•	Amenities: Terrace, garden, etc.

	•	Price: Target variable for prediction.


## Column	Description
property_id	Unique identifier for each property
price	Target variable: property price (in EUR)
location	Property location
type	Type of property (house, apartment, etc.)
bedrooms	Number of bedrooms
surface_area	Property surface area (sqm)
garden	Whether property has a garden
terrace	Whether property has a terrace
...	Other relevant features
The dataset is saved in a CSV file as immoweb_data.csv.

## Requirements
To run this project, you’ll need:

	•	Python 3.12

	•	Jupyter Notebook 

	•	Required packages in requirements.txt

## Libraries:
	•	pandas: Data manipulation

	•	scikit-learn: Machine learning

	•	xgboost: Gradient boosting algorithm

	•	matplotlib and seaborn: Data visualization


# Data Preprocessing

## 1.Prepare the dataset:

Place the properties.csv file in the project directory if not already present.
Data Preprocessing
Handle Missing Values:

## Categorical NaNs are replaced with "MISSING".
Numeric NaNs are filled with the median or mean values.
Encoding:

## One-hot encoding for categorical variables.
Dummy variables are prefixed with fl_.
Feature Scaling:

## Numeric features are standardized for uniformity across the model.
Feature Engineering:

## Generate additional features, if necessary, such as area per room or price per square meter.

2. Train Model
Select Algorithms:

Linear Regression, Decision Trees, Random Forest, and other regression algorithms.

3. Make Predictions

Once the model is trained, use it to make predictions:

python predict.py --input path/to/input.csv --output path/to/output.csv

This will take an input CSV file with new property listings and output predictions to the specified path.


4.Model Evaluation


The primary evaluation metric for this project is the Mean Absolute Error (MAE) for continuous variables like price. Other metrics include:

	•	Mean Squared Error (MSE)

	•	R-squared (R²)



Save the best-performing model for use in the API.


# Pipeline Description


	1.	Data Collection to collect property data from ImmoWeb.

	2.	Preprocessing: Clean and preprocess data in preprocess_data.py.

	3.	Model Training: Train various models and tune hyperparameters using train_model.py.

	4.	Prediction: Use predict.py to generate predictions on new data.

	5.	Evaluation: Evaluate model performance using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.





Use evaluate_model.py to evaluate a trained model:



python evaluate_model.py --model path/to/model.pkl --test_data path/to/test.csv



# Linear regression for a real estate website like Immoweb in Python, i would follow these steps. The goal is likely to predict property prices based on features like square footage, number of rooms, location, etc. Below is a basic guide using Python’s pandas, scikit-learn, and matplotlib libraries.


THen linear regression, you need to define your features (independent variables) and the target (dependent variable, usually the property price).

1.Define features (X) and target (y)

2.Then Split Data into Training and Test Sets.Splitting the data helps in assessing the model’s performance on unseen data.

3.Train the Linear Regression Model

use scikit-learn’s LinearRegression to train the model.

4.Initialize the model

model = LinearRegression()

5.Train the model

6: Make Predictions and Evaluate the Model.Once the model is trained, you can evaluate its performance using the test data.

 Calculate Mean Squared Error and R-squared score

7.(Optional) Visualize Results

If you want to see how predictions compare to actual values, you can plot them.

This is a basic framework. To improve results, you can experiment with feature engineering, more advanced models, or regularization techniques.



# The Decision Tree Regressor 
is a type of supervised learning algorithm used for regression tasks, meaning it predicts continuous values (as opposed to classification tasks where the output is categorical). It belongs to the family of decision tree algorithms and operates by splitting the data into subsets based on certain conditions, creating a tree-like structure.

# How it Works
The algorithm recursively partitions the dataset by choosing the most significant feature at each step, aiming to reduce the prediction error. Each split is chosen to minimize a particular metric (like mean squared error or mean absolute error), making the model progressively better at predicting the target variable. The final result is a model that resembles a tree structure, where:

Nodes represent features of the data,
Branches represent decisions or conditions, and
Leaves represent the final predicted value.

# When we use Decision Tree Regression

Decision Tree Regression is useful when you want to make continuous predictions (e.g., predicting a price or a measurement) and the underlying data may have complex, non-linear relationships. Here’s when it’s particularly beneficial:
1.	Handling Non-linear Data: Decision trees can capture non-linear relationships in data, making them a good choice when linear models (like linear regression) are insufficient. They split the data into regions based on different conditions, creating a piecewise function that can model complex patterns.
2.	Feature Interaction: Decision trees can automatically capture interactions between features, as they split based on feature thresholds that best fit the target variable. This means they don’t require you to specify interactions in advance.
3.	Minimal Preprocessing: Decision trees don’t require feature scaling, such as normalization or standardization, and they handle categorical data well (though categorical features may need to be preprocessed in specific libraries like scikit-learn).
4.	Interpretable Models: A decision tree model is relatively easy to interpret visually by examining the splits and decision nodes, which can help understand which features are important and how they influence predictions.
Examples of Uses
•	Real Estate Pricing: Estimating property prices based on complex feature interactions, like location, size, and amenities.
•	Healthcare: Predicting patient outcomes based on age, medical history, and other features where interactions matter.
•	Manufacturing: Estimating production yield or defect rates based on conditions in different stages of the production process.


# XGBoost (Extreme Gradient Boosting) is a powerful machine learning algorithm optimized for speed and performance, commonly used for structured/tabular data. Here’s a guide on how to use XGBoost in Python.


## 1. Linear Regression
Strengths: Linear regression is simple, interpretable, and computationally efficient. It works well with linear relationships and provides coefficients that indicate the influence of each feature on the price.
Weaknesses: Linear regression assumes a linear relationship between features and the target variable, which may not capture the complexity of real estate prices well. It's also sensitive to outliers, which can distort predictions.
Best Use Case: If your data has a strong linear relationship and few outliers, linear regression might perform well. However, it often struggles with complex, non-linear relationships like those in real estate data.
## 2. Decision Tree Regressor
Strengths: Decision trees are more flexible and can capture non-linear relationships in data. They are interpretable to a degree because they split data into "if-then" decisions, which can show important splits based on feature values.
Weaknesses: Decision trees can easily overfit on training data, especially when deep trees are allowed. This overfitting can result in poor generalization to new data unless regularized or limited by depth.
Best Use Case: Decision trees are useful when the data is somewhat complex but not overly large. They are often used as a base for more advanced methods like Random Forests or XGBoost.
## 3. XGBoost Regressor
Strengths: XGBoost is a powerful, flexible ensemble method that combines many decision trees to make predictions. It is known for its high accuracy and ability to handle non-linear relationships. XGBoost also provides regularization options to prevent overfitting, making it robust on large and complex datasets.

Weaknesses: XGBoost is computationally intensive and may take longer to train compared to simpler models. Tuning hyperparameters can also be complex, as it has many configuration options.
## Best Use Case: XGBoost is particularly effective when you have a large, complex dataset with non-linear relationships, as is often the case in real estate. It’s also a good choice if you are aiming for high accuracy and are willing to trade off interpretability.

Model Comparison Using Evaluation Metrics
To decide which model is best, you should compare them using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) on both the training and test sets. Here’s what to look for:

Lower RMSE or MAE: This indicates more accurate predictions.
Generalization (Train vs. Test Performance): Check that the model performs consistently across train and test sets to avoid overfitting.
General Recommendation:

### XGBoost is likely to perform best given its robustness with non-linear relationships, complex interactions, and ability to manage large datasets.
Decision Tree could work if interpretability is a priority and if you're dealing with fewer data points, but it may overfit.
Linear Regression may perform well only if the relationships are primarily linear; however, in most real estate data, non-linear models (like XGBoost) tend to perform better due to the complex factors influencing prices.

If high accuracy and model performance are top priorities, XGBoost is likely the best choice for your real estate price prediction task. Let me know if you'd like help with metrics calculations for a direct comparison.



# The conclusion the best score for the model was in XG boost 

# Score for linear Regression Train Score is 50.8
# Score for linear Regression Test Score is 50.5

# Score for Decision Train Score  is 97.9
# Score for Decision Test Score  is 92.8


# Score for XGboost Train Score is 98.7
# Score for XGboost Test Score  is 94.9

