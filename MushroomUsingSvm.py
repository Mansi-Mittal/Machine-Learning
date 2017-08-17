import numpy as np
import pandas as pd
from sklearn import preprocessing,cross_validation,svm,metrics,neighbors 
p=pd.read_csv("mushrooms.csv")

for name in p.columns:
	p[name]=preprocessing.LabelEncoder().fit_transform(p[name])

Y=np.array(p["class"])
X=np.array(p.drop(["class","veil-type"],1))

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.3)
model=neighbors.KNeighborsClassifier()

model.fit(X_train,Y_train)


predicted=model.predict(X_test)
predicted=np.round(predicted)

accu=model.score(X_test,Y_test)
print(accu)
print(metrics.classification_report(Y_test,predicted))