# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import libraries required for data processing and machine learning.
2.Load and preprocess the dataset using Pandas.
3.Separate features (X) and target (y) variables.
4.Split the dataset into training and testing sets.
5.Train the model, predict results, and evaluate accuracy. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:RAJAPRABU.S 
RegisterNumber:212225240113  
*/
import pandas as pd from sklearn.model_selection import train_test_split from sklearn.tree import DecisionTreeClassifier from sklearn.preprocessing import LabelEncoder from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

data = pd.read_csv("C:/Users/acer/Downloads/Employee.csv")

print("Dataset Sample:") print(data.head())

le = LabelEncoder()

data['Departments '] = le.fit_transform(data['Departments ']) data['salary'] = le.fit_transform(data['salary'])

X = data.drop("left", axis=1) y = data["left"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Predicted Values:", y_pred) print("Accuracy of the model:", accuracy) print("\nClassification Report\n",classification_report(y_test,y_pred))
```

## Output:
<img width="1010" height="702" alt="image" src="https://github.com/user-attachments/assets/085d7689-7dba-4628-b9fe-bd99d5e759a8" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
