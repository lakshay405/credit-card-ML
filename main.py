import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the credit card dataset into a Pandas DataFrame
df_credit = pd.read_csv('/content/credit_data.csv')

# Displaying the first 5 rows of the dataset
print(df_credit.head())

# Dataset information
print(df_credit.info())

# Checking for missing values in each column
print(df_credit.isnull().sum())

# Distribution of legitimate transactions (Class 0) and fraudulent transactions (Class 1)
print(df_credit['Class'].value_counts())
# 0 --> Normal Transaction
# 1 --> Fraudulent Transaction

# Separating the data for analysis
legit = df_credit[df_credit.Class == 0]
fraud = df_credit[df_credit.Class == 1]
print(legit.shape)
print(fraud.shape)

# Statistical measures of transaction amounts
print(legit.Amount.describe())
print(fraud.Amount.describe())

# Comparing the mean values for both types of transactions
print(df_credit.groupby('Class').mean())

# Creating a balanced dataset with equal samples of both classes
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
print(new_dataset.tail())
print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())

# Separating features (X) and target variable (Y)
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Initializing and training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Predicting on training data and calculating accuracy
Y_train_pred = model.predict(X_train)
training_accuracy = accuracy_score(Y_train_pred, Y_train)
print('Accuracy on Training data: ', training_accuracy)

# Predicting on test data and calculating accuracy
Y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test_pred, Y_test)
print('Accuracy on Test data: ', test_accuracy)
