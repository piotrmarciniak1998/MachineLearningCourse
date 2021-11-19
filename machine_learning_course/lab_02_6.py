from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

battery = pd.read_csv("battery_problem_data.csv")
dataset = np.transpose(battery.values, (1, 0))

data = dataset[0].reshape(len(dataset[0]), 1) * 100
target = dataset[1] * 100
data = data.astype(int)
target = target.astype(int)

# print(data)
# print(target)

X, y = data, target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# reg = linear_model.LinearRegression()
# reg.fit(X_train, y_train)
# y_test_out = reg.predict(X_test)

# plt.plot(X_test, y_test, 'bo', X_test, y_test_out, 'ro')
# plt.show()

reg = DecisionTreeRegressor(max_depth=5)
reg.fit(X_train, y_train)
y_test_out = reg.predict(X_test)

X_test = X_test.astype(float)
y_test = y_test.astype(float)
y_test_out = y_test_out.astype(float)

X_test /= 100
y_test /= 100
y_test_out /= 100

print("Mean absolute error: " + str(mean_absolute_error(y_test, y_test_out)))
print("Mean squared error: " + str(mean_squared_error(y_test, y_test_out)))
print("R2 score: " + str(r2_score(y_test, y_test_out)))

plt.plot(X_test, y_test, 'bo', X_test, y_test_out, 'ro')
plt.show()

