# Customer Churn Prediction Using Artificial Neural Network (ANN)

## Overview of this project
Customer churn prediction is a crucial task for businesses, especially in the telecom industry. The goal is to identify factors that contribute to customer churn and build a deep learning model to predict it. In this project, I employed an Artificial Neural Network (ANN) for customer churn prediction. The model's performance was evaluated using precision, recall, and F1-score metrics. This is an end to end machine learning project.

## Prerequisites
- Python 3.x
- Libraries used: pandas, matplotlib, numpy, scikit-learn, seaborn, tensorflow/keras

## Dataset
Dataset was gotten from : https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The data set includes information about:

- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

## I:
1. Loaded the data 
2. Performed data cleaning by dropping the 'customerID' column as it is not useful. I converted 'TotalCharges' to numeric, and handled blank strings.
3. Performed some data visualization for churned and non-churned customers.
4. Performed data preprocessing:
   - Replaced 'No internet service' and 'No phone service' with 'No'.
   - Converted 'Yes' and 'No' to 1 and 0.
   - Performed one-hot encoding for categorical columns.
   - Scaled numerical features using Min-Max scaling.
5. Did Train-Test Split
6. Built and trained an ANN model using TensorFlow/Keras.
7. Evaluated the model on the test set and printed classification report.
8. Visualized the confusion matrix using seaborn.
9. Checked the model performance using accuracy, precision, recall, and F1-score metrics.

## Conclusion
This project demonstrates the use of an Artificial Neural Network for customer churn prediction. The model achieved a certain level of accuracy and provides insights into the precision, recall, and F1-score for both churned and non-churned customers.