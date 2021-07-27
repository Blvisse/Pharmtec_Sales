"""
Module to read data from dvc storage

"""

import pandas as pd
import logging
import dvc.api
import mlflow 


logging.basicConfig(filename='../logs/DVC.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)



class DVCDATA:

    def __init__(self):
        logging.debug("Initializing Class")
       

    def get_data(self,exp,datapath,repository,version):
        logging.debug("Initializing get_data() function")
        

        #raise a file not found error if this occurs to break the whole loop from runing
        
        try:
            with mlflow.start_run():
                print("Fetching data from Repo ... \n")

            
                dataurl=dvc.api.get_url(path=datapath,repo=repository,rev=version)
                mlflow.set_experiment(exp)        
                print("Converting into dataframe \n")
                data=pd.read_csv(dataurl)

                logging.debug("Logging parameters to mlflow")
                print("Logging data deatails to mlflow \n")
                mlflow.log_param("Data url",dataurl)
                mlflow.log_param("Data version", version)
                mlflow.log_param("Input rows", data.shape[0])
                mlflow.log_param("Input columns",data.shape[1])

                print("Done....")
                logging.debug("Suuccesfully ran function.")
                logging.info("Data URL ----- {} ----- Data Verion ------ {} ----- Data rows ------{} ------ Data columns ----- {} --- \n".format(dataurl,version,data.shape[0],data.shape[1]))

                mlflow.end_run()
            
                return data, dataurl,version

        except  Exception as e:
            logging.error("Program run into error... \n ")
            logging.error("Error class {} ".format(e.__class__))
            print ("The program ran into an error ... \n ")
            print("Error details {} ".format(e))
            raise FileNotFoundError ("The files weren't Retrived")


        

if (__name__== '__main__'):
    instance=DVCDATA()
    data,dataurl,version=instance.get_data('data/AdSmartABdata.csv','https://github.com/Blvisse/ABtesting','cbf35f6e43a99698ba53718ed8a8c8da8f3da722')
    print(data)

        
