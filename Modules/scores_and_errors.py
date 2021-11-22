from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from Modules.setup_logger import setup_logger
import os


class Score():
    def __init__(self):
        self.folder = './logs/'
        self.filename = 'score_and_errors.txt'
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = setup_logger("Validation metrices", self.folder+self.filename)
        
    
    def evaluation_r2_score(self,act,pred):
        """
        Methode: evaluation_r2_score
        Description: Caculate the r2 score
        Input: actual values, predicted values
        Output: r2
        on falure: log error

        Version: 1.0
        """
        try:
            self.log_object.info('Calculating r2 score')
            r2_sc = r2_score(act,pred)
            self.log_object.info("Calculated r2 score: --Done")
            return r2_sc
        except Exception as e:
            self.log_object.info("Error in calcualating r2 socre " + str(e))      
    
    
    def mae(self,act,pred):
        """
        Methode : mae
        Description: Caculate the mean absolute error
        Input: actual values, predicted values
        Output: mae
        on falure: log error

        Version: 1.0
        """
        try:
            self.log_object.info("Calculating mean absolute error") 
            self.log_object.create_log_file()
            mae = mean_absolute_error(act,pred)
            
            self.log_object.info("Calculated mean absolute error: --Done")
         
            return mae
        except Exception as e:
            self.log_object.info("Error in calcualating mean absolute error " + str(e))

    
    def rmse(self,act,pred):
        """
        Methode: rmse
        Description: Caculate the root mean squared error
        Input: actual values, predicted values
        Output: rmse
        on falure: log error

        Version: 1.0
        """
        try:
            self.log_object.info("Calculating root mean squared error")           
            mse = mean_squared_error(act,pred)
            rmse = np.sqrt(mse)
            self.log_object.info("Calculated root mean squared error: --Done") 

            return rmse
        except Exception as e:
            self.log_object.info("Error in calcualating root mean squared error " + str(e)) 