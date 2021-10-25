from sklearn.svm import SVC
import numpy as np


class Weather:
    data = None
    target = None

    def __init__(self, temp_list, time_list, rain_list, target_list):
        data = []
        for i in range(len(temp_list)):
            data.append([temp_list[i], time_list[i], rain_list[i]])
        self.data = np.array(data)
        self.target = np.array(target_list)


temp =   [2,   18,  -15, 22,  16]
time =   [12,  16,  0,   12,  22]
rain =   [0.5, 0.2, 1.0, 0.3, 0.1]
target = [0,   1,   0,   1,   0]

weather = Weather(temp, time, rain, target)

X = weather.data
y = weather.target

clf = SVC()
clf.fit(X, y)

print(clf.predict([[15, 14, 0.2]]))
