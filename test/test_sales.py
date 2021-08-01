
  
import unittest
import pandas as pd
import sys,os

sys.path.append(os.path.abspath(os.path.join('../..')))

import logging 
import dvc.api

logging.basicConfig(filename='../logs/test.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)


data=pd.read_csv('../data/Merged.csv')
class TestsAB(unittest.TestCase):


    
    
    
    def test_checkNulls(self):

        logging.debug(" ---------------- initializing Test Check Nulls ---------------- \n")
        print (" =============== Running Check Nulls Test ================= \n ")
        try: 
            self.assertEqual(data['Sales'].isnull().sum(),0)
            print("Passed null test \n")
            logging.info(" ---------------- Passed test ----------------")
        except :
            logging.error(" ---------------- Failed Test ---------------- ")
            print("Failed Test ")
         

    def test_duplicates(self):

        logging.debug("============== Initalizing Test Duplicates ============== ") 
        print (" ========== Running Duplicates Test ================= \n")
        try:
            numduplicates =len(data)-len(data.drop_duplicates())
            self.assertEqual(numduplicates,0)
            print("Passed duplicates test \n")
            logging.info(" ========== Passed test ============= ")
        except:
            logging.error(" ---------------- Failed Test ---------------- ")
            print("Failed Test ")

    # def test_no_sales(self):

        # for stores in data['Sales']:
        #     try:
        #         self.assertEqual(stores,0)
        #         print (" ===== Passed test ======= ")
        #     except :
        #         print (" ===== Failed test =====")

         




    
    


   

if __name__ == '__main__':
    unittest.main()


