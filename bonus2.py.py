import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn .metrics import mean_squared_error
from sklearn.model_selection import train_test_split , StratifiedKFold
from sklearn import datasets
from sklearn.metrics import accuracy_score , confusion_matrix 
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier


"""
iris = datasets.load_iris()
x = iris.data
y = iris.target
"""
data= pd.read_csv(r"F:\3rd comp\second term\AI\sections\bonous2\aps_failure_training_set_processed_8bit.csv")
x = data.iloc[:, 1: len(list(data.columns))-1]
y = data.iloc[: , 0]
"""
for i in range(0 , len(y)-1):
    if y[i] >0 :
        y[i] = 1
    elif y[i] <0 :
        y[i] = 0
        
"""

predicted_list = []
algorithms = [
        "LogisticRegression",
        "GradientBoostingClassifier",
        "RandomForestClassifier",
        "XGBClassifier",
        "DecisionTreeClassifier"
        
        
        
        ]


y[y>0] = 1
y[y<0] = 0

skf = StratifiedKFold(n_splits=5)
predicted = np.zeros(y.shape[0])

for train , test in skf.split(x , y):
    x_train = x.iloc[train , :]
    x_test = x.iloc[test , :]
    y_train = y[train]
    y_test = y[test]
    logreg = LogisticRegression()
    logreg.fit(x_train , y_train)
    re = logreg.predict(x_test)
    predicted[test] = re
print("using LogisticRegression is :" , accuracy_score(y , predicted) * 100)
predicted_list.append(accuracy_score(y , predicted) * 100)

conf = confusion_matrix(y , predicted)

for train , test in skf.split(x , y):
    x_train = x.iloc[train , :]
    x_test = x.iloc[test , :]
    y_train = y[train]
    y_test = y[test]
    gradient_boost = GradientBoostingClassifier()
    gradient_boost.fit(x_train , y_train)
    re = gradient_boost.predict(x_test)
    predicted[test] = re
print("using GradientBoostingClassifier is :" , accuracy_score(y , predicted) * 100)
predicted_list.append(accuracy_score(y , predicted) * 100)

conf = confusion_matrix(y , predicted)

for train , test in skf.split(x , y):
    x_train = x.iloc[train , :]
    x_test = x.iloc[test , :]
    y_train = y[train]
    y_test = y[test]
    random_forest = RandomForestClassifier()
    random_forest.fit(x_train , y_train)
    re = random_forest.predict(x_test)
    predicted[test] = re
print("using RandomForestClassifier is :" , accuracy_score(y , predicted) * 100)
predicted_list.append(accuracy_score(y , predicted) * 100)

conf = confusion_matrix(y , predicted)

for train , test in skf.split(x , y):
    x_train = x.iloc[train , :]
    x_test = x.iloc[test , :]
    y_train = y[train]
    y_test = y[test]
    xgbClassifier = XGBClassifier(n_estimators=135 , learning_rate = .08 , )
    xgbClassifier.fit(x_train , y_train)
    re = xgbClassifier.predict(x_test)
    predicted[test] = re
print("using XGBClassifier is " , accuracy_score(y , predicted) * 100)
predicted_list.append(accuracy_score(y , predicted) * 100)

conf = confusion_matrix(y , predicted)

  
for train , test in skf.split(x , y):
    x_train = x.iloc[train , :]
    x_test = x.iloc[test , :]
    y_train = y[train]
    y_test = y[test]
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(x_train , y_train)
    re = decision_tree.predict(x_test)
    predicted[test] = re
print("using DecisionTreeClassifier is :" , accuracy_score(y , predicted) * 100)
predicted_list.append(accuracy_score(y , predicted) * 100)

conf = confusion_matrix(y , predicted)   


    
    

    
    





#2






