"""Data Separation"""
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Modules.setup_logger import setup_logger
import pickle

class Splitting_And_Scaling:
    def __init__(self,df):
        self.df = df
        self.folder = './logs/'
        self.filename = 'splitting_and_scaling.txt'
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = setup_logger("splitting_and_scaling",self.folder+self.filename)

    def split(self):
        try:
            self.log_object.info('Start splitting dataset into training and test set')
            X = self.df.drop(columns='rate')
            Y = self.df[['rate']]
            
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state = 353)
            self.log_object.info('Training and test set split completed')
            return x_train, x_test, y_train, y_test
        
        except Exception as e:
            self.log_object.info("Error in splitting dataset into training and test set "+ str(e))
            

    def scaling(self):
        try:
            self.log_object.info('Start scaling the data')
            x_train, x_test, y_train, y_test = self.split()

            std = StandardScaler()  # creating instance of StandardScaler class
            
            x_train_std = std.fit_transform(x_train)

            x_test_std = std.transform(x_test)           # unseen data
            self.log_object.info('Data scaling completed')
            
            with open('std_scaler.pkl', 'wb') as file:
                pickle.dump(std, file)
            
            self.log_object.info('scaling model saved')
            return x_train_std, x_test_std, y_train, y_test
        
        except Exception as e:
            self.log_object.info("Error in scaling the data " + str(e))
            
            