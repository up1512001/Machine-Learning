# work with " The Boston Housing Dataset "
#

from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston = datasets.load_boston()
X = boston.data
y = boston.target

#print(X)
#print(y)
#print(X.shape)
#print(y.shape)

# algorithm
l_reg = linear_model.LinearRegression()

#plt.scatter(X.T[5], y)
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = l_reg.fit(X_train,y_train)

pred = model.predict(X_test)

print("Prediction:", pred)
print("R^2 value:", l_reg.score(X,y))
print("Coefficient:", l_reg.coef_)
print("Intersept:",l_reg.intercept_)
