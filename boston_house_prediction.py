# Using regression models to predict housing prices on the Boston housing dataset (inbuilt with Sklearn)
# Link to Data:  https://www.kaggle.com/c/house-price-prediction-with-boston-housing-dataset/data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score

# Encode labels
data_train = pd.read_csv('HousingData.csv')
# print(data_train.dtypes)

# print(data_train.shape)
# shape of the dataframe ie no. of rows and columns


# checking for any duplicates in the data
print(data_train.duplicated().sum())

# checking for any null values in the data
# print(data_train.isnull().sum())

# removing null values
data_train.dropna(axis=0, inplace=True)

# EDA with Data Visualization
#
# Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to
# discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics
# and graphical representations.
#
# Data visualization is the graphical representation of data in order to interactively and efficiently convey
# insights to clients, customers, and stakeholders in general.

# getting the information of dataframe such as no. of entries,data columns,non-null count,data types,etc
# print(data_train.info)

# checking for statistical summary such as count,mean,etc. of numeric columns
print(data_train.describe())

# checking for any outliers in the data
sns.boxplot(data=data_train, orient='h', palette='Set2')

# singular data visual rep
sns.boxplot(x=data_train['TAX'])

sns.boxplot(x=data_train['AGE'])

sns.boxplot(x=data_train['MEDV'])

Q1 = data_train.quantile(0.25)
Q3 = data_train.quantile(0.75)
IQR = Q3 - Q1
# print(IQR)

# Correlation matrix
#
# Correlation coefficients quantify the association between variables or features of a dataset. These statistics are
# of high importance for science and technology, and Python has great tools that you can use to calculate them.
# SciPy, NumPy, and Pandas correlation methods are fast, comprehensive, and well-documented.
#
# The correlation matrix can be used to estimate the linear historical relationship between the returns of multiple
# assets. You can use the built-in . corr() method on a pandas DataFrame to easily calculate the correlation matrix.
# Correlation ranges from -1 to 1.

data_train.corr()
# finding the correlation between different variables/features

# Heatmap¶
#
# A heatmap is a graphical representation of data in which data values are represented as colors. That is,
# it uses color in order to communicate a value to the reader. This is a great tool to assist the audience towards
# the areas that matter the most when you have a large volume of data.


train_corr = data_train.corr()
f, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(train_corr, cmap='viridis', annot=True)
# plt.title("Correlation between features", weight='bold', fontsize=18)
# plt.show()

# plotting the heatmap for different features


# From the above heatmap ,we can conclude that :
# >> Price(MEDV) greatly depends upon features RM (positively correlated) and LSTAT (negatively correlated).
# >> Also, the features AGE and DIS are negatively correlated with each other.
#
# i.e if a house is older then Weighted distances to five Boston employment centers decreases. Similarly, other such
# pairs are "DIS - NOX", "DIS - INDUS" and "LSTAT - RM". >> And features TAX and RAD are positively correlated with
# each other.
#
# i.e if Tax increases, accessibility to radial highways also increases.

X = data_train.drop('MEDV', axis=1)
y = data_train['MEDV']

y = np.round(data_train['MEDV'])

# Apply SelectKBest class to extract top 5 best features
bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# Concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns

print(featureScores.nlargest(5, 'Score'))  # print 5 best features
model.fit(X, y)

print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_train)

# calculating the accuracies
print("Training Accuracy :", model.score(X_train, y_train) * 100)
print("Testing Accuracy :", model.score(X_test, y_test) * 100)

print("Model Accuracy", r2_score(y, model.predict(X)) * 100)
