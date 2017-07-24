# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
STEP 1:
-------
Loading data
"""
## Import load_iris function datasets module
from sklearn.datasets import load_iris

# save in bunchs
iris = load_iris()

# store feature matrix in X
X = iris.data
 
# store response vector in y
y = iris.target

#print the shape of X and y
print(X.shape)
print(y.shape)
print iris.feature_names
print iris.target
print iris.target_names

"""
STEP 2:
------
Using sklearn for Kneighbor Classifier
"""
from sklearn.neighbors import KNeighborsClassifier
"""
Instantiate ---> make an instance 
Estimator ---->  is scikit-learn for model
"""
knn = KNeighborsClassifier(n_neighbors=1)
#print(knn)

"""
STEP 3:
------
fitting the model with data 
"""
knn.fit(X,y)

"""
STEP 4:
-------
predict the response for new observation
"""
print knn.predict([3,4,6,5])

print "We can predict multiple observation"

X_new = [[4,7,4,2],[6,8,3,1]]
print knn.predict(X_new)

"""
Using a different value for K
"""
#Instantiate the model (Using the value k = 5)
knn = KNeighborsClassifier(n_neighbors = 5)

#fit the model with data 
knn.fit(X,y)

#predict the response for new observation
print knn.predict(X_new)
"""
Using a different classification model
"""
#import the class 
from sklearn.linear_model import LogisticRegression

#Instantiate the model(using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X,y) 

#predict the response
print logreg.predict(X_new)





























