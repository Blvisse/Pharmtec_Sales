"""
This script cleans data 

"""

import pandas as pd
import numpy as np
import mlflow 
import logging


logging.basicConfig(filename='../logs/cleanData.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)


class CLEAN:

    def __init__(self,data):
        
        self.data=data

    def calculate_null(self):
        
            #This function calculates the number of missing values in the dataset

        try:
            logging.debug("Calculating the missing values")

            sumMissing=self.data.isna().sum()
            percentageMissing=sumMissing/len(self.data)*100

            #create a dataframe to present the findings
            logging.info("Creating Dataframe")

            missingValues=pd.DataFrame(data=[sumMissing,percentageMissing])
            missingValues=missingValues.T
            missingValues.columns=['Total Missing','Percentage Missing']

            logging.info("Missing Values calculated")
            print("The report of missing values is as follows")

            return missingValues

        except Exception as e:
            logging.error("Error message {} ".format(e.__class__))
            print("The following error occured{} ".format(e.__class__))

    def dropDuplicates(self):

        # This function looks for duplicates and drops them 
        try:
            logging.debug("Launching duplicates search")
            print("Droppping duplicates\n")
            data=self.data.drop_duplicates()
            dupCount=len(self.data)-len(self.data.drop_duplicates())
            print ("There are {} duplicates in the dataset\n".format(dupCount))
            logging.info("Number of duplicates in the datset are {} ".format(dupCount))

            print("Done dropping duplicates! \n")

            return data 

        except Exception as e:
            logging.info("An erro has occured")
            logging.error("The follocing error occured {} ".format(e.__class__))
            print("The following error occured {} ".format(e.__class__))


    def valueCounts(self,columns):
        # function to calculate the number of records per each unique value
        try:
            logging.debug("Value counts of {} column".format(columns))
            count=self.data[columns].value_counts()
            countDF=pd.DataFrame(data=[count])
            count=count.T 

            return count
        except Exception as e:
            logging.info("An erro occured")
            logging.error("The following error occured {} ".format(e.__class__))
            print("The following error occured {} ".format(e.__class__))

