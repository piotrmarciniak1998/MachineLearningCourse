from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC


wine = datasets.load_wine()

X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LinearSVC()
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
