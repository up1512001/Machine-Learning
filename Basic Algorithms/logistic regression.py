from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:,3:]
y = (iris.target == 2).astype(np.int)

# print(iris.keys())
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename']
# print(iris['DESCR'])

clf = LogisticRegression()
clf.fit(X, y)
exa = clf.predict([[2.6]])
print(exa)

X_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(X_new)
# print(y_prob)
# print(X_new)
# plt.plot(X_new, y_prob[:, 1], "g-", label="virginica")
# plt.show()

