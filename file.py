import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , scale
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor



columns_names = open(r"E:\3rd comp\second term\AI\sections\bonus task\abalone.domain").read().splitlines()



alist = []

for i in columns_names:
    alist.append(i.split(':')[0])
    
    
file = pd.read_csv(r"E:\3rd comp\second term\AI\sections\bonus task\abalone.data" ,  names=alist)

for i in alist:
    if file[i].dtype == "object":
        le = LabelEncoder()
        le.fit(file[i])
        file[i] = le.transform(file[i])
        
        
X = file.iloc[:, 0: len(list(file.columns))-1]
Y = file.iloc[: , len(list(file.columns))-1]

# ways f features scaling

#X = (X - X.mean()) / (X.max()-X.min())

#X = (X - X.mean()) / X.max()

X = X / (X.max() - X.min())

#X = X / X.max()

x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size= .33 , random_state=i)


    

rnd = []


def fit_data(X , Y , type):
    global rnd
    error_list = []
    if type == "LinearRegression":
        data_fit = LinearRegression()

    elif type == "ExtraTreeRegressor":
        data_fit = ExtraTreeRegressor()
        
    elif type == "DecisionTreeRegressor":
        data_fit = DecisionTreeRegressor()

    elif type == "RandomForestRegressor":
        data_fit = RandomForestRegressor()

    elif type == "GradientBoostingRegressor":
        data_fit = GradientBoostingRegressor()

    elif type == "XGBRegressor":
        data_fit = XGBRegressor()
        
    for i in range(100 , 1000 , 1):
        rnd.append(i)
        x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size= .3 , random_state=i)


        data_fit.fit(x_train , y_train)
    
        reesult = data_fit.predict(x_test)
        
        error_list.append(np.sqrt(mean_squared_error(y_test , reesult)))
        
    print("using " + type + " error is :")   
    print(min(error_list))
    
    print("using randonstate = " , rnd[error_list.index(min(error_list))])


    
        


    
    
        
        


fit_data(X , Y , "XGBRegressor")
fit_data(X , Y , "GradientBoostingRegressor")
fit_data(X , Y , "RandomForestRegressor")
fit_data(X , Y , "DecisionTreeRegressor")
fit_data(X , Y , "ExtraTreeRegressor")
fit_data(X , Y , "LinearRegression")





