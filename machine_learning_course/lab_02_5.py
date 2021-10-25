from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

battery = pd.read_csv("battery_problem_data.csv")
dataset = np.transpose(battery.values, (1, 0))
data = dataset[0].reshape(len(dataset[0]), 1)
target = dataset[1].reshape(len(dataset[0]), 1)

print(data)
print(target)

X, y = data, target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC()
clf.fit(X_train, y_train)

print(clf.predict(X_test))
