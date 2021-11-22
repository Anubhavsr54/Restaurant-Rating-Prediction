from Modules.setup_logger import setup_logger
from Modules.data_loader import DataGetter
from Modules.data_transformation import DataTransformation
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import json


class Preprocessor():
    """
    Class for preprocessing the data
    Description: This class contain all necessary method to preprocessthe data.
    return: Preprocessd data saved on local

    """
    def __init__(self):
        self.log = setup_logger("preprocessor_log", "logs/preprocessor.log") # logger
        self.data_getter = DataGetter() # data getter object
        self.data_transformation = DataTransformation() # data transformation object



    def drop_columns(self, data, col_list):
        """
            Method: drop_columns

            Descriptions: this method will passed column from data

            Input: data: pandas dataframe
                col_list: list of columns to be dropped

            Output: data: pandas dataframe after dropping columns

            On error: raise error, log error on log files

            Version: 1.0
        """
        try:
            self.log.info("Dropping columns from data")
            data.drop(col_list, axis=1, inplace=True) # drop columns with inplace True, for reflecting on original data
            self.log.info("Columns dropped successfully")
            return data # return data after dropping columns
        except Exception as e:
            self.log.error("Error occured while dropping columns from data " + str(e))
            raise Exception("Error occured while dropping columns from data " + str(e))


    
    def check_vif(self, data):
        """
        Methode: check_vif
        Description: This method will check VIF(Variance Inflation_factor) on data
                    and will drop those columns have VIF value above predefined threshold
        Input: data: pandas dataframe
        Output: data: pandas dataframe after removing columns if VIF value is above threshold for columns.
        
        On error: raise error, log error on log files

        Version: 1.0
        """
        try:
            self.log.info("Checking VIF on data")
            vif = pd.DataFrame() # create empty dataframe to store vif values
            vif["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])] # calculate vif values
            vif["features"] = data.columns # add column name to dataframe
            self.log.info("VIF DataFrame Created")

            # remove columns with vif value above threshold
            threshold = 5.0 # can be varied.
            vif_filter = vif[vif["VIF Factor"] < threshold] # filtering dataframe with vif value less than threshold
            col_to_drop = vif_filter["features"].tolist() # list of columns to be dropped
            data = self.drop_columns(data, col_to_drop) # droping columns from data
            self.log.info("VIF checked successfully")
            return data # return data after removing columns

        except Exception as e:
            self.log.error("Error occured while checking VIF on data " + str(e))
            raise Exception("Error occured while checking VIF on data " + str(e))


    def drop_null(self, data):
        """
        Method: drop_null

        Descriptions: this method will drop null values from data

        Input: data: pandas dataframe
        Output: data: pandas dataframe after dropping null values
        
        On error: raise error, log error on log files

        Version: 1.0
        """
        try:
            self.log.info("Dropping null values from data")
            data.dropna(inplace=True) # drop null values with inplace True, for reflecting on original data
            self.log.info("Null values dropped successfully")
            return data # return data after dropping null values
        except Exception as e:
            self.log.error("Error occured while dropping null values from data " + str(e))
            raise Exception("Error occured while dropping null values from data " + str(e))

    
    def drop_duplicates(self, data):
        """
        Method: drop_duplicate

        Descriptions: this method will drop duplicate values from data

        Input: data: pandas dataframe
        Output: data: pandas dataframe after dropping duplicate values
        
        On error: raise error, log error on log files

        Version: 1.0
        """
        try:
            self.log.info("Dropping duplicate values from data")
            data.drop_duplicates(inplace=True) # drop duplicate values with inplace True, for reflecting on original data
            self.log.info("Duplicate values dropped successfully")
            return data # return data after dropping duplicate values
        except Exception as e:
            self.log.error("Error occured while dropping duplicate values from data " + str(e))
            raise Exception("Error occured while dropping duplicate values from data " + str(e))



    def label_encoding(self, data, cat_col):
        """
        Method: label_encoding
        Description: This methode will use sklearn LabelEncoder
                    to encoding categorial features.
                    also it will store encoding label with respective feature label.
        Input: data: pandas dataframe.
                cat_col: (str) single categorical feature column name
        Output: data: pandas dataframe after encoding categorical features.
                label_encode_dict: (dict) dictionary of label encoding with respective feature label.

        On Error: raise error, log Error.

        Version: 1.0
        """
        try:
            self.log.info("Label encoding categorical features.. Started")
            # creating instance of LabelEncoder
            le = LabelEncoder()
            # encoding categorical features
            data[cat_col] = le.fit_transform(data[cat_col]) # replacing categorical features with encoded values.
            self.log.info("Label encoding categorical features.. Compleated")
            # storing label encoding with respective feature label
            # creating empty dictionary to store label encoding
            label_encoding_dict = dict() # empty dictionary
            for num, label in enumerate(le.classes_):
                label_encoding_dict[label] = num # storing label encoding with respective feature label
            self.log.info("Label encoding categorical features.. Stored")
            return data, label_encoding_dict # return data after encoding categorical features and label encoding dictionary

        except Exception as e:
            self.log.error("Error occured while label encoding categorical features " + str(e))
            raise Exception("Error occured while label encoding categorical features " + str(e))