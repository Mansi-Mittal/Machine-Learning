import numpy as py
from sklearn import datasets
from sklearn import tree
import pydotplus

p=pd.read_csv("glass.csv")

Y=np.array(p["Type"])
X=np.array(p.drop(["Type"],1))
X=np.array(p.drop(["RI"],1))

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.2)

model=tree.DecisionTreeClassifier()
model.fit(X_train,Y_train)

dot_data = dot_data = tree.export_graphviz(model, out_file=None,
                         feature_names=p.feature_names,
                         class_names=p.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("glass.pdf")