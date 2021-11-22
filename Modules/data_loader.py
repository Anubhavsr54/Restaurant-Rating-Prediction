from Modules.setup_logger import setup_logger
import pandas as pd
import json

class DataGetter():
    """
    Class for loading data from local and saving data to local
    """
    def __init__(self):

        # setting up looger
        self.log = setup_logger(logger_name="DataGeter_LOg",
                                log_file="logs/DataGeter.log")

        # self.data_path = "./zomato.csv"

    def data_getter(self, data_path):
        """
        Method: data_getter
        Description: loading csv data from local and return pandas dataframe

        Input: data_path
        Output: pandas datframe

        On Error: log error and raise error

        Version: 1.0
        """
        try:
            self.log.info("Loading the data form local...")
            data = pd.read_csv(data_path)
            self.log.info("Data load successfully")
            # returning the data 
            return data   # returning pandas dataframe.
        except Exception as e:
            self.log.error("Error on Loading data")
            raise e


    def data_saved(self, data, data_path):
        """
        Method: data_saved
        Description: saving the data to local

        Input: data_path: path to save the data
                        : data: data to save as pandas dataframe
        Output: Saved scv file to data_path loaction 

        On Error: log error and raise error

        Version: 1.0
        """
        try:
            self.log.info("Saving the data to local...")
            data.to_csv(data_path, index=False) # saving the data to local
            self.log.info("Data saved successfully")
        except Exception as e:
            self.log.error("Error on Saving data")
            raise e


    def save_json(self, label_encoding_dict, file_name):
        """
        Methode: save_json
        Description: This method will save dictionary as json file
        Input: dict_: (dict) dictionary to be saved
        Output: None

        On Error: raise error, log Error.

        Version: 1.0
        """
        try:
            self.log.info("Saving dictionary as json file")
            with open(file_name, 'w') as f:
                json.dump(label_encoding_dict, f, indent=4) # dumping dictionary as json file
            self.log.info("Dictionary saved as json file")
        except Exception as e:
            self.log.error("Error occured while saving dictionary as json file " + str(e))
            raise Exception("Error occured while saving dictionary as json file " + str(e))

    
    def load_json(self, file_name):
        """
        Methode: load_json
        Description: This method will load dictionary from json file
        Input: file_name: (str) name of the json file
        Output: (dict) dictionary loaded from json file

        On Error: raise error, log Error.

        Version: 1.0
        """
        try:
            self.log.info("Loading dictionary from json file")
            with open(file_name, 'r') as f:
                dict_ = json.load(f) # loading dictionary from json file
            self.log.info("Dictionary loaded from json file")
            return dict_
        except Exception as e:
            self.log.error("Error occured while loading dictionary from json file " + str(e))
            raise Exception("Error occured while loading dictionary from json file " + str(e))