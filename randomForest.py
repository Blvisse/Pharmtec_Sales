import os
import sys
import logging
module_path =os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"\\scripts")

if module_path not in sys.path:
    sys.path.append(module_path+"\\model")

logging.basicConfig(filename='logs/model.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)


logging.debug('------ Importing Libraries ----')


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
import dvc.api
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler 
from xgboost import XGBRegressor  
from sklearn.tree import DecisionTreeRegressor 
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

PATH=pathlib.Path(__file__).parent
DATA_PATH=PATH.joinpath("./data").resolve

dpath=PATH.joinpath("Merged.csv")

#We have created a pipeline we want to carry out preprocessing

#first lets initialize the classes

print("------- Calling Data Class to fetch Data from dvc ----------\n")
try:
    
    logging.debug("Initalizing Data Class")
    # instance=DVCDATA()
    # data,dataurl,version=instance.get_data('Holiday Tech','data/Merged.csv','https://github.com/Blvisse/Pharmtec_Sales','maindata-v2')
    # url=dvc.api.get_url(path='data/Merged.csv',repo='https://github.com/Blvisse/Pharmtec_Sales',rev='maindata-v2')
    print("Getting data from dvc...")
    data=pd.read_csv('data/Merged.csv')
    logging.debug("Successfully acquired data from dvc")
    print(" -------------------------------- Acquired data from DVC --------------------------------")
    

    print(data)

except Exception as e:
    logging.error("Failed to get data")
    print(" The system has failed to retrieve the data from dvc ")


#######################   Creating a preprocessing pipeline    #########################################
print (" ---- Initializing preprocessing pipeline ------ \n")
logging.debug (" Initializing preprocessing pipeline")

## we need to drop some columns that might not be of use to the model
dropcols=['Date','Customers','Open','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','PromoInterval']
data.drop(labels=dropcols,inplace=True,axis=1)

print("-------------------------------- Dropped Columns --------------------------------")
logging.info("Prepared Data")


ordcols=['StoreType','Assortment']
numcols=['Store','Promo','CompetitionDistance','CompetitionOpen','Promo2','Promo2Open','IsPromo2Month','Day','Month','Year','Week']



data['StateHoliday']=data['StateHoliday'].astype('category')

data['StateHoliday']=data['StateHoliday'].cat.codes

oe=OrdinalEncoder()
data[ordcols]=oe.fit_transform(data[ordcols])




print(data.info())

# ### create a scaler and also split data ##
print("Getting data features ")
y=data['Sales']
X=data.drop('Sales',axis=1)

##split data
logging.debug("Spliting Data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


print(" ---------------- Entering Pipeline  -------------------------------- \n")


model=Pipeline([('Scaler ',StandardScaler()),            ('XGB Boost', XGBRegressor())],
             verbose=True)
print(" -------------------------------- Initializing XGB Regressor -------------------------------- \n")
logging.debug (" -------------------------------- Initializing XGBRegressor -------------------------------- \n ")
lrPipeline=Pipeline([('Scaler ',StandardScaler()), ('XG Boost', XGBRegressor())],verbose=True)

print(" -------------------------------- Initializing Random Forest Regressor --------------------------------\n")
logging.debug (" -------------------------------- Initializing Random Forest Regressor --------------------------------")
fPipeline=Pipeline([('Scaler ',StandardScaler()),('RandomForestRegressor',RandomForestRegressor())],verbose=True)

print(" -------------------------------- Initializing Random DecisionTree Regressor --------------------------------\n")
logging.debug (" -------------------------------- Initializing Decision Tree Regressor -------------------------------- \n")
descPipline=Pipeline([('Scaler ',StandardScaler()), ('Decision Tree', DecisionTreeRegressor())],verbose=True)

pipelines=[lrPipeline,descPipline]
countnames=0
for mod in pipelines:
    logging.debug(" -------------------- Creating Experiment -------------- \n")
    print("---------------- Creating mlflow Experiment... --------------------- \n")
    mlflow.set_experiment("Model Training")
    with mlflow.start_run():
        mod.fit(X_train,y_train)
        prediction=mod.predict(X_test)
        mse=mean_squared_error(y_test,prediction)
        mae=mean_absolute_error(y_test,prediction)

        trainScores =mod.score(X_train,y_train)
        testScores=mod.score(X_test,y_test)
        
        print(" ---------- Logging parameters to mlflow -----")
        mlflow.log_param(" Model ", mod)
        mlflow.log_metric("Mean Squared Error",mse)
        mlflow.log_metric("Mean Absolute Error",mae)
        mlflow.log_metric("Training Variance",trainScores)
        mlflow.log_metric("Test Variance",testScores)

        print(" -------------------------------- Creating plot --------------------------------")
        plt.figure(figsize=(10,9))
        sns.scatterplot(x=y_test,y=prediction,hue=prediction)
        plt.savefig("Model.png")
        mlflow.log_artifact("Model.png")

        print(" -------- Writing files -------- ")
        filenames=["metric.txt","metric2.txt","metric3.txt"]
        with open(filenames[countnames] ,'w') as outfile:
            outfile.write("Training Variance: %2.1f%%\n"%trainScores)
            outfile.write("Test Variance: %2.1f%%\n"%testScores)
            outfile.write("Mean Squared: %2.1f%%\n"%mse)
            outfile.write("Mean Absolute Error: %2.1f%%\n"%mae)
        
        
        # image formatting
        axis_fs = 18 #fontsize
        title_fs = 22 #fontsize
        sns.set_theme()
        
        resultsDF=pd.DataFrame(list(zip(y_test,prediction)),columns=["true","pred"])
        ax = sns.scatterplot(x="true", y="pred",data=resultsDF)
        ax.set_aspect('equal')
        ax.set_xlabel('True Sales quality',fontsize = axis_fs) 
        ax.set_ylabel('Predicted Sales', fontsize = axis_fs)#ylabel
        ax.set_title('Residuals', fontsize = title_fs)

        # Make it pretty- square aspect ratio
        ax.plot([1, 10], [1, 10], 'black', linewidth=1)
  
        picnames=["residuals.png","residuals2.png","residuals3.png"]
        
        plt.tight_layout()
        plt.savefig(picnames[countnames],dpi=120) 
        countnames+=1
        



        print(mean_squared_error(y_test,mod.predict(X_test)))

# models=Pipeline[('Scaler ',StandardScaler()),('XGB Boost', XGBRegressor()),('Scaler ',StandardScaler()),('RandomForest', XGBRegressor()),('Scaler ',RandomForestRegressor()),('Decision Tree', DecisionTreeRegressor())]

# lrPipeline.fit(X_train,y_train)
# print(mean_squared_error(y_test,lrPipeline.predict(X_test)))













