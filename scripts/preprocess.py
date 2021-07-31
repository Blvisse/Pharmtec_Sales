"""
This module carries out preprocessing on data

"""


import pandas as pd
import numpy as np
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler,OrdinalEncoder




logging.basicConfig(filename='../logs/preprocess.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)

class PREPROCESS:

    def __init__(self):
        logging.debug("Initializing Class")
        

    def cleanOutliers(self,data,cols):
        logging.debug("Accessed Clean Outliers Function")
        print("Getting numerical features..\n")


        try:
            print ("Detecting outliers..\n")
            for col in cols:
                print("Calculating limits \n")
                upperLimit=data[cols].mean()+3*data[cols].std()
                lowerLimit=data[cols].mean()-3*data[cols].std()

                data=data[(data[cols]<upperLimit)&(data[cols]>lowerLimit)]

            print ("Sorted outliers...\n")
            return data 


        except Exception as e:
            logging.info("An error occured")
            logging.error("The following error occured {} ".format(e.__class__))
            print("The following error occured {} ".format(e.__class__))
    

    def labelEncoding(self,data,cols):
        logging.debug("-------------------------------- Acessing label Encoding Functoin --------------------------------")
        print("------- Acessing label encoding Functoin -----------")
        

        try: 
            logging.debug("encoding columns")
            print("--- Encoding columns ---")
            data[cols]=data[cols].astype('category')
            data[cols]=data[cols].cat.codes
            
            return data 

        except Exception as e:
            logging.info("An error occured")
            logging.error("The following error has occured: {} ".format(e.__class__))

    
    
    def ordinalEncodings(self,data,cols):
        logging.debug("------- Accessing the ordinal function ------------")
        print("----- Accessing the ordinal function ------------")

        logging.debug("Initializng OrdinalEncoder")
        oe=OrdinalEncoder()

        print("Encoding columns")
        try:
            
            data[cols]=oe.fit_transform(data[cols])
            
            return data
        except Exception as e:
            logging.info("An error occured")
            logging.error("The following error has occured: {} ".format(e.__class__))



            
   
   
    def scalingFeature(self,data):
        
        logging.debug("Accessing the scaling features function")
        print("--- Scaling features ---\n")
        try:
            logging.debug("Initializing Standard Scaler")
            ss=StandardScaler()

            print("--- Scaling Data ---")
            data=ss.fit_transform(data)
            print("--- Scaling Data ---")

            return data
        
        except Exception as e:
            logging.info("An error occured")
            logging.error("The following error occured {} ".format(e.__class__))
            print("The following error occured {} ".format(e.__class__))


          
