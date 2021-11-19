from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
# from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X, y = datasets.load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, stratify=y)
X_train = X_train[['sepal length (cm)', 'sepal width (cm)']]
X_test = X_test[['sepal length (cm)', 'sepal width (cm)']]

clf_std = Pipeline([
    ('skaler', StandardScaler()),
    ('svc', SVC())
])

clf_std.fit(X_train, y_train)

plt.scatter(np.array(X)[:, 0],
            np.array(X)[:, 1])

# plot_decision_regions(np.array(X_test)[0:2], np.array(y_test), clf=clf, legend=1)
plt.title("Iris sepal features")
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.show()

