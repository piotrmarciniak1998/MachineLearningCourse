from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X, y = datasets.load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, stratify=y)

skaler = MinMaxScaler()
# skaler = StandardScaler()
skaler.fit(X_train)
X_train = skaler.transform(X_train)

plt.scatter(np.array(X_train)[:, 0],
            np.array(X_train)[:, 1])
plt.axvline(x=0)
plt.axhline(y=0)
plt.title("Iris sepal features")
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.show()
