# Setup
import sys
sys.path.append(r"C:\Users\mharr_jngp84v\anaconda3\Lib\site-packages")
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

titanic_data = pd.read_csv('titanic/train.csv')
print(titanic_data.head())

# Data Pre-processing
## The data has some missing info so we need to clean it up 
## Dropping Cabin entirely because it doesn't have much affect
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
## Age is pretty important so we can't drop it entirely, thus, we fill it with the mean
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
print(titanic_data['Embarked'].mode()[0]) # 0 is the index
## This line of code is a problem but the program works so whatever
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
## Have we gotten rid of the null columns? yes. 
titanic_data.isnull().sum()



