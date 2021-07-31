
  
import unittest
import pandas as pd
import sys,os

sys.path.append(os.path.abspath(os.path.join('../..')))


import dvc.api

data=pd.read_csv('data/Merged.csv')
class TestsAB(unittest.TestCase):


    
    
    
    def test_checkNulls(self):

        

        self.assertEqual(data['Sales'].isnull().sum(),0)

    
    


   

if __name__ == '__main__':
    unittest.main()


