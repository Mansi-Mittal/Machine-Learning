import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing,cross_validation,svm,metrics
p=pd.read_csv("glass.csv")

#p.info()
#print(p.describe())

Y=np.array(p["Type"])
X=np.array(p.drop(["Type"],1))
X=np.array(p.drop(["RI"],1))


X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.2)
model=GaussianNB()
model.fit(X_train,Y_train)
predicted=model.predict(X_test)
predicted=np.round(predicted)
accu=model.score(X_test,Y_test)
print(accu)
print(metrics.classification_report(Y_test,predicted))

