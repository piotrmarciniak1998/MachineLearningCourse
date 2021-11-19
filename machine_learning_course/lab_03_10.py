from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X, y = datasets.load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, stratify=y)
#X_train = X_train[['sepal length (cm)', 'sepal width (cm)']]
#X_test = X_test[['sepal length (cm)', 'sepal width (cm)']]

clfs = ["LogisticRegression", "SVC", "DecisionTreeClassifier", "RandomForestClassifier"]

results = dict()

for clf in clfs:
    pipe = Pipeline([
        ('scale', MinMaxScaler()),
        (clf, globals()[clf]())
    ])
    pipe.fit(X_train, y_train)
    # print(f"{clf}: {pipe.score(X_test, y_test)}")
    results[clf] = pipe.score(X_test, y_test)

print(results)

