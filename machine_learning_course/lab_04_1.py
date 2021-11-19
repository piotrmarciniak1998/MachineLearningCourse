import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import impute
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from mlxtend.plotting import plot_decision_regions


diabetes = pd.read_csv("../diabetes.csv")
diabetes["class"].loc[diabetes["class"]=="tested_positive"] = 1
diabetes["class"].loc[diabetes["class"]=="tested_negative"] = 0
diabetes["class"] = diabetes["class"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(diabetes.drop(["class"], axis=1),
                                                    diabetes["class"],
                                                    random_state=42,
                                                    stratify=diabetes["class"],
                                                    test_size=0.25)

clfs = ["LogisticRegression", "SVC", "DecisionTreeClassifier", "RandomForestClassifier", "KNeighborsClassifier"]

results = dict()

for clf in clfs:
    pipe = Pipeline([
        ('scale', MinMaxScaler()),
        (clf, globals()[clf]())
    ])
    pipe.fit(X_train, y_train)
    results[clf] = pipe.score(X_test, y_test)

for a in ["plas", "pres", "skin", "insu", "mass"]:
    X_train[a].loc[X_train[a]==0] = np.NaN

for clf in clfs:
    pipe = Pipeline([
        ('imputer', impute.KNNImputer()),
        ('scale', MinMaxScaler()),
        (clf, globals()[clf]())
    ])
    pipe.fit(X_train.dropna(), y_train[X_train.notna().all(axis=1)])
    results[clf + "(Imputer)"] = pipe.score(X_test.dropna(), y_test[X_test.notna().all(axis=1)])

for item in results:
    print(f"{item}: {results[item]}")

zscore = abs((diabetes - diabetes.mean()) / diabetes.std())
diabetes = diabetes.loc[~(zscore>=3).any(axis=1)]
'''
clf = IsolationForest()
clf.fit(diabetes[["mass", "plas"]])
plot_decision_regions(np.array(diabetes[["mass", "plas"]]),
                      np.array(clf.predict(diabetes[["mass", "plas"]])),
                      clf)
plt.show()
'''


from sklearn import svm, model_selection
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'],
                                                    random_state=42,
                                                    stratify=iris['target'],
                                                    test_size=0.25)

parameters = {
    'kernel': ('linear', 'rbf', 'sigmoid'),
    'C':[1, 10, 30,]
}

clf = GridSearchCV(svm.SVC(), parameters, cv=10)
clf.fit(X_train, y_train)

pvt = pd.pivot_table(
    pd.DataFrame(clf.cv_results_),
    values='mean_test_score',
    index='param_kernel',
    columns='param_C'
)

ax = sns.heatmap(pvt)
plt.show()


import pickle


with open("model.pickle", "wb") as file:
    pickle.dump(clf.best_estimator_, file)

print(clf.best_estimator_.score(X_test, y_test))