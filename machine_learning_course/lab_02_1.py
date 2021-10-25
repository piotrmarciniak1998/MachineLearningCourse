from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
import matplotlib.pyplot as plt


digits = datasets.load_digits()

'''
elements = 5
fig, axs = plt.subplots(len(digits.target_names), elements)
for nr in range(len(digits.target_names)):
    for i in range(elements):
        axs[nr][i].imshow(digits.images[digits.target==nr][i], cmap='gray_r')
        axs[nr][i].axis('off')
plt.show()
'''

X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X1 = [X_train[0]]
y1 = [0]

clf = SVC()
clf.fit(X1, y1)

print(clf.predict(X1))

