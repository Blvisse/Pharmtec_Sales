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
from sklearn.decomposition import PCA 
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.linear_model import LinearRegression 
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

from sklearn.model_selection import GridSearchCV

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

print(" -------------------------------- Initializing XGB Regressor with PCA-------------------------------- \n")
logging.debug (" -------------------------------- Initializing XGBRegressor with PCA -------------------------------- \n ")
lrpcPipeline=Pipeline([('Scaler ',StandardScaler()), ('Principal Component Analysis',PCA(n_components=2)), ('XG Boost', XGBRegressor())],verbose=True)


print(" -------------------------------- Initializing Random Forest Regressor --------------------------------\n")
logging.debug (" -------------------------------- Initializing Random Forest Regressor --------------------------------")
fPipeline=Pipeline([('Scaler ',StandardScaler()),('RandomForestRegressor',RandomForestRegressor())],verbose=True)

print(" -------------------------------- Initializing Random DecisionTree Regressor --------------------------------\n")
logging.debug (" -------------------------------- Initializing Decision Tree Regressor -------------------------------- \n")
descPipline=Pipeline([('Scaler ',StandardScaler()), ('Decision Tree', DecisionTreeRegressor())],verbose=True)

print(" -------------------------------- Initializing Random DecisionTree Regressor with PCA --------------------------------\n")
logging.debug (" -------------------------------- Initializing Decision Tree Regressor with PCA -------------------------------- \n")
descpcPipline=Pipeline([('Scaler ',StandardScaler()),('Principal Component Analysis',PCA(n_components=2)), ('Decision Tree',DecisionTreeRegressor())],verbose=True)


print(" -------------------------------- Initializing Random DecisionTree Regressor --------------------------------\n")
logging.debug (" -------------------------------- Initializing Decision Tree Regressor -------------------------------- \n")
linearpcPipline=Pipeline([('Scaler ',StandardScaler()), ('Principal Component Analysis',PCA(n_components=2)),('Linear Regressor', LinearRegression())],verbose=True)

print(" -------------------------------- Initializing Random DecisionTree Regressor --------------------------------\n")
logging.debug (" -------------------------------- Initializing Decision Tree Regressor -------------------------------- \n")
linearPipline=Pipeline([('Scaler ',StandardScaler()), ('Linear Regression', LinearRegression())],verbose=True)




pipelines=[lrPipeline,lrpcPipeline,descPipline,descpcPipline,linearPipline,linearpcPipline]
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
        
        # if mod.named_steps != mod['Linear Regression']:
        print(mod.named_steps)
        print(" ---------- Logging parameters to mlflow -----")
        mlflow.log_param(" Model ", mod.named_steps)
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
        filenames=["metricxgb.txt","metricxgbpca.txt","metricdec.txt","metricdecpca.txt","metriclineae.txt","metriclineaepca.txt"]
        with open(filenames[countnames] ,'w') as outfile:
            outfile.write("Training Variance: %2.1f%%\n"%trainScores)
            outfile.write("Test Variance: %2.1f%%\n"%testScores)
            outfile.write("Mean Squared: %2.1f%%\n"%mse)
            outfile.write("Mean Absolute Error: %2.1f%%\n"%mae)
            mlflow.log_artifact(filenames[countnames])
        
        
        # image formatting
        axis_fs = 18 #fontsize
        title_fs = 22 #fontsize
        sns.set_theme()
        print("Plotting graph")
        resultsDF=pd.DataFrame(list(zip(y_test,prediction)),columns=["true","pred"])
        ax = sns.scatterplot(x="true", y="pred",data=resultsDF)
        ax.set_aspect('equal')
        ax.set_xlabel('True Sales quality',fontsize = axis_fs) 
        ax.set_ylabel('Predicted Sales', fontsize = axis_fs)#ylabel
        ax.set_title('Residuals', fontsize = title_fs)

        # Make it pretty- square aspect ratio
        ax.plot([1, 10], [1, 10], 'black', linewidth=1)
  
        picnames=["predxgb.png","predxgbpca.png","preddec.png","preddecpca.png","predlineae.png","predclineaepca.png"]

        
        plt.tight_layout()
        plt.savefig(picnames[countnames],dpi=120) 
        mlflow.log_artifact(picnames[countnames])

        countnames+=1
        
        print(" -------------------------------- Done --------------------------------")



        print(mean_squared_error(y_test,mod.predict(X_test)))

###### Hyper Parameter Tune our models #################
print("-------- Hyper Parameter Tune Decision Tree -------")
print(descPipline.named_steps['Decision Tree'])
dtparameters=[{'splitter':["best","random"],
            'max_depth' : [4,7,9],
           'min_samples_leaf':[1,2,3],
           'min_weight_fraction_leaf':[0.1,0.2],
           'max_features':["auto","log2","sqrt",None],
          'max_leaf_nodes':[None,10,20]
           }]
xgparams=[{'max_depth':[8,9,10],
            'min_child_weight':[2,3,4],
            'gamma':[0,1,2]
            }]
jobs=-1
print(" -------------------------------- Creating grid search --------------------------------")
logging.debug("Initializing Grid Search...")
dtSearch=GridSearchCV(estimator=descPipline.named_steps['Decision Tree'],param_grid=dtparameters,scoring='neg_mean_squared_error',cv=3,verbose=3,n_jobs=jobs)
xgSearch=GridSearchCV(estimator=lrPipeline.named_steps['XG Boost'],param_grid=xgparams,scoring='neg_mean_squared_error',cv=3,verbose=3)
grids=[dtSearch]
grid_dict={0: 'Desc Tree'}
print('Performing model optimizations...')
for idx,gs in enumerate(grids):
    with mlflow.start_run():
        mlflow.set_experiment("Hyper Parameter Tuning")
        mlflow.log_param("Model Search",grid_dict[idx])
        print('\nEstimator: %s' % grid_dict[idx])	
        # Fit grid search
        # print(gs.get_params().keys())
        gs.fit(X_train, y_train)
        # gs.fit(X_train, y_train)

        print('Best params: %s' % gs.best_params_)
        mlflow.log_param("Best Metrics",gs.best_params_)
        # # Best training data accuracy
        print('Best training accuracy: %.3f' % gs.best_score_)
        mlflow.log_metric("Best Score",gs.best_score_)
        print("Best Estimator",gs.best_estimator_)
        mlflow.log_param("Feature Importance",gs.best_estimator_.feature_importances_)
        importance=gs.best_estimator_.feature_importances_
        labels=X_train.columns
        feature=pd.DataFrame(list(zip(labels,importance)),columns=['feature','importance'])

        # image formatting
        axis_fs = 18 #fontsize
        title_fs = 22 #fontsize
        sns.set(style="whitegrid")
        ax = sns.barplot(x="importance", y="feature", data=feature)
        ax.set_xlabel('Importance',fontsize = axis_fs) 
        ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
        ax.set_title('Decision Tree \nfeature importance', fontsize = title_fs)

        plt.tight_layout()
        plt.savefig("feature_importance.png",dpi=120)
        mlflow.log_artifact("feature_importance.png")
        mlflow.sklearn.log_model(gs.best_estimator_,"Best Model")

# dtSearch.fit(X_train,y_train) 
# dtSearch.best_params
# dtSearch.best_scores













