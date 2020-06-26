import pandas
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

pd = pandas.read_csv("./data/bank-additional-full.csv", sep=";")
pd = pd.drop(["duration"], axis=1)
label = LabelEncoder()
fields = ['job', 'marital', 'education',
          'default', 'housing', 'loan', 'contact',
          'month', 'day_of_week', 'poutcome']
for field in fields:
    pd[field] = label.fit_transform(pd[field])
y = label.fit_transform(pd["y"])
pd = pd.drop(["y"], axis=1)
# Split the data
x_train, x_test, y_train, y_test = train_test_split(pd, y, test_size=0.3)
# Tree_model
clf_tree = DecisionTreeClassifier()     # using default criterion and splitter
clf_tree = clf_tree.fit(x_train, y_train)
y_predict_tree = clf_tree.predict(x_test)

accuracy_score_tree = accuracy_score(y_test, y_predict_tree)
precision_score_tree = precision_score(y_test, y_predict_tree)
recall_score_tree = recall_score(y_test, y_predict_tree)
roc_auc_tree = roc_auc_score(y_test, y_predict_tree)
# KNN_model
clf_KNN = KNeighborsClassifier()
clf_KNN = clf_KNN.fit(x_train, y_train)
y_predict_KNN = clf_KNN.predict(x_test)

accuracy_score_KNN = accuracy_score(y_test, y_predict_KNN)
precision_score_KNN = precision_score(y_test, y_predict_KNN)
recall_score_KNN = recall_score(y_test, y_predict_KNN)
roc_auc_KNN = roc_auc_score(y_test, y_predict_KNN)
# print(roc_auc_KNN,roc_auc_tree)
