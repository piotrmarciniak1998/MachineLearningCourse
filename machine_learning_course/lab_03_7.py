from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X, y = datasets.load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, stratify=y)

clf_minmax = Pipeline([
    ('skaler', MinMaxScaler()),
    ('svc', RandomForestRegressor())
])

clf_std = Pipeline([
    ('skaler', StandardScaler()),
    ('svc', RandomForestRegressor())
])

clf_minmax.fit(X_train, y_train)
print(clf_minmax.score(X_test, y_test))

clf_std.fit(X_train, y_train)
print(clf_std.score(X_test, y_test))

