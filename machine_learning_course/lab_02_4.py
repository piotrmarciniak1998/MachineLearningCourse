from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
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

clf = SVC()
clf.fit(X_train, y_train)

plot_confusion_matrix(clf, X_test, y_test)
plt.show()