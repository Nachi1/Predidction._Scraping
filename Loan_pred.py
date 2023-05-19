# Loan Prediction Dataset
# https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

loan_data = pd.read_csv('train_loan_pred.csv')

loan_data.dropna()
loan_data['Gender'] = [1 if a == 'Male' else 0 for a in loan_data['Gender']]
loan_data['Married'] = [1 if a == 'Yes' else 0 for a in loan_data['Married']]
loan_data['Education'] = [1 if a == 'Graduate' else 0 for a in loan_data['Education']]
loan_data['Self_Employed'] = [1 if a == 'No' else 0 for a in loan_data['Self_Employed']]
loan_data['Property_Area'] = [1 if a == 'Rural' else 2 if a == 'Semiurban' else 0 for a in loan_data['Property_Area']]
loan_data['Dependents'] = [4 if a == '3+' else 0 for a in loan_data['Dependents']]
loan_data['Loan_Status'] = [1 if a == 'Y' else 0 for a in loan_data['Loan_Status']]

# print(loan_data.head())

loan_data = loan_data.dropna()
# print(loan_data['Loan_Status'].value_counts())
# print(loan_data['Gender'].value_counts())
# print(loan_data['Education'].value_counts())
# print(loan_data['Married'].value_counts())
# print(loan_data['Self-Employed'].value_counts())
# print(loan_data['Property_Area'].value_counts())
# print(loan_data['Dependents'].value_counts())

# train / Split
X = loan_data.drop(['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_data.Loan_Status

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33)

# Comparison of loan status based on gender
# print(loan_data.groupby("Gender")['Loan_Status'].value_counts())
# print(sns.countplot(data=loan_data, x="Loan_Status"))

# sns.countplot(data=train["Loan_Status"], x=train["Gender"], hue=train["Loan_Status"])

# Comparison of loan approvals based on married status
# loan_data.groupby("Married")["Loan_Status"].value_counts()

# print(loan_data.dtypes)
# model training
model = RandomForestClassifier()

# fit data
model.fit(train_X, train_Y)

# Prediction
pred = model.predict(test_X)

print(accuracy_score(test_Y, pred))
