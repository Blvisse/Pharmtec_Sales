"""

This module conducts univariate anlysis on the features

"""

import pandas as pd
import numpy as np
import logging
from scipy import stats



logging.basicConfig(filename='../logs/univ.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)




class UNIV:

    def __init__(self,data):
        logging.debug("Initializing class")
        self.data=data

    def calculateMetrics(self,col):
        #convert the column to a numpy array to perform the metric evaluation
        try:
            print("---- Reading data ----")
            
            
            logging.debug("--- Accessing DataFrame ---")
            print("--- Done ---\n")
            print('--- Calculating the univariate metrics of the columns {} --- '.format(col))
            
            logging.debug(" ---- Calculating metrics ---- ")
            print("Converting the data into numpy arrays")
            nmArray=self.data[col].values
            print("Done!")
            print("Calculating mean....")        
            mean=np.mean(nmArray)
            print ("Calculating Mode...")
            mode=stats.mode(nmArray)[0]
            print("Calculating Median...")
            median=np.median(nmArray)
            print ("Calculating skeweness...")
            skew=stats.skew(nmArray)
            print ("Calculating kurtosis...")
            kurtosis=stats.kurtosis(nmArray)
            print ("Calculating standard deviation....")
            std=np.std(nmArray)
            print ("Calculating variance ...")

            var=np.var(nmArray)

            print("Done !")
            
            columns=['Mean','Mode','Median','Skew','Kurtosis','Standard deviation','Variance']

            statsdf=[{'Mean':mean,'Mode':mode,'Median':median,'Skew':skew,'Kurtosis':kurtosis,'Standard deviation':std,'Variance':var}]
            df=pd.DataFrame(data=statsdf,columns=columns)
            df=df.T
            df.columns=['Analysis Values']

            logging.debug(" ---- Finalized Calculating the Metrics ---- ")

            return df

        except Exception as e:
            logging.error(" --- Error message {} ---- ".format(e.__class__))
            print("The following error occured{} ".format(e.__class__))

    def calculateDispersion(self,col):
        # nmArray=self.data[self.col].to_numpy()
        
        try:

            logging.debug("--- Accessing DataFrame ---")
            print("Calculating dispersion stats for {}".format(col))

            logging.debug(" ---- Calculating metrics ---- ")
            nmArray=self.data[col].values
            
            Q1=np.quantile(nmArray,.25)
            Q2=np.quantile(nmArray,.50)
            Q3=np.quantile(nmArray,.75)

            print("Calculating Standard Deviation")

            std=np.std(nmArray)

            print ("Calculating Inter Quartile Range")

            IQR=Q3-Q1

            print ("Caclulating Max ")
            maxVal=np.max(nmArray)
            minVal=np.min(nmArray)

            print("Done....\n ")
            
            dispdf=[{'Q1':Q1,'Q2':Q2,'Q3':Q3,'Std deviation':std,'IQR':IQR,'Max Value':maxVal,'Min Value':minVal}]
            print("Creating DataFrame")
            df=pd.DataFrame(data=dispdf)
            df=df.T
            df.columns=['Dispersion Values']
            logging.debug(" ---- Finalized Calculating the Metrics ---- ")

            return df
        
        except Exception as e:

            logging.error(" --- Error message {} ---- ".format(e.__class__))
            print("The following error occured{} ".format(e.__class__))





         

    

    
        
