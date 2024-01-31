# Customer Churn Prediction

## This is an end to end machine learning project on customer churn prediction using Artificial Neural Networks (ANN)  to measure why customers are leaving a telecom business. I built a deep learning model to predict the churn and use precision,recall, f1-score to measure performance of my model.

Dataset was gotten from : https://www.kaggle.com/datasets/blastchar/telco-customer-churn

About Dataset
Context
"Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]

Content
Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

The data set includes information about:

Customers who left within the last month – the column is called Churn
Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
Demographic info about customers – gender, age range, and if they have partners and dependents


# Customer Churn Prediction Using Artificial Neural Network (ANN)

## Overview of this project
Customer churn prediction is a crucial task for businesses, especially in the telecom industry. The goal is to identify factors that contribute to customer churn and build a deep learning model to predict it. In this project, I employed an Artificial Neural Network (ANN) for customer churn prediction. The model's performance was evaluated using precision, recall, and F1-score metrics.

## Prerequisites
- Python 3.x
- Libraries used: pandas, matplotlib, numpy, scikit-learn, seaborn, tensorflow/keras

## I:
1. Loaded the data 
2. Performed data cleaning by dropping the 'customerID' column as it is not useful. I converted 'TotalCharges' to numeric, and handled blank strings.
3. Performed some data visualization for churned and non-churned customers.
4. Performed data preprocessing:
   - Replaced 'No internet service' and 'No phone service' with 'No'.
   - Converted 'Yes' and 'No' to 1 and 0.
   - Performed one-hot encoding for categorical columns.
   - Scaled numerical features using Min-Max scaling.
5. Did Train-Test Split:
   - Split the data into training and testing sets.

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
   ```

6. Build and Train the Model:
   - Build an ANN using TensorFlow/Keras.

   ```python
   # Code for building and training the model
   ```

7. Evaluate the Model:
   - Evaluate the model on the test set and print classification report.

   ```python
   # Code for evaluating the model
   ```

8. Visualize Confusion Matrix:
   - Visualize the confusion matrix using seaborn.

   ```python
   # Code for visualizing the confusion matrix
   ```

9. Model Performance Metrics:
   - Calculate accuracy, precision, recall, and F1-score.

   ```python
   # Code for calculating performance metrics
   ```

## Conclusion
This project demonstrates the use of an Artificial Neural Network for customer churn prediction. The model achieved a certain level of accuracy and provides insights into the precision, recall, and F1-score for both churned and non-churned customers. Further fine-tuning and optimization can be performed to enhance the model's predictive capabilities.