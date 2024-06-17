# credit-card-ML
Credit Card Fraud Detection using Logistic Regression
This project focuses on detecting fraudulent credit card transactions using machine learning, specifically Logistic Regression. The goal is to build a model that can accurately classify transactions as either legitimate or fraudulent based on various transaction attributes.

Dataset
The dataset (credit_data.csv) includes information about credit card transactions, such as transaction amount and a binary class label indicating whether the transaction is legitimate (Class 0) or fraudulent (Class 1).

Workflow
Data Loading and Preprocessing:

Load the credit card transaction data from CSV file into a Pandas DataFrame (df_credit).
Perform initial exploration by displaying the first few rows, checking dataset information, and identifying any missing values.
Exploratory Data Analysis (EDA):

Explore the distribution of transaction classes (Class 0 for normal transactions and Class 1 for fraudulent transactions).
Analyze statistical measures (mean, count, etc.) of transaction amounts for both types of transactions to understand differences.
Handling Imbalanced Data:

Address the class imbalance by creating a balanced dataset with an equal number of samples from both classes (legitimate and fraudulent transactions).
Model Training and Evaluation:

Separate features (X) and the target variable (Y) from the balanced dataset.
Split the data into training and testing sets using train_test_split.
Initialize and train a Logistic Regression model to classify transactions.
Evaluate the model's performance using accuracy score on both training and testing sets.
Libraries Used
numpy and pandas for data manipulation and analysis.
sklearn for model selection (LogisticRegression), evaluation (train_test_split, accuracy_score), and handling imbalanced data.
Conclusion
This project demonstrates the application of Logistic Regression for detecting credit card fraud by training a model on a balanced dataset derived from real-world transaction data. The model achieves a high accuracy in distinguishing between legitimate and fraudulent transactions, providing a robust solution for fraud detection in financial transactions.
