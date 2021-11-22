from Modules.setup_logger import setup_logger
from Modules.data_loader import DataGetter

class DataTransformation():
    """
    Class for tranforming data
    """
    def __init__(self):
        self.log = setup_logger("Datatranformation.log", 'logs/data_transformation.log') # set up logger


    def rename_columns(self, data):
        """
        Method for renaming columns
        Description: this method will chnae the column names to the required names

        :param columns_list: dataframe, old_columns_list: list, new_columns_list: list
        :return: dataframe after renaming the columns names

        On Failure: Raise ValueError, log error in application logger

        Version: 1.0
        """
        try:
            self.log.info('Renaming columns Started')
            data = data.rename(columns={'approx_cost(for two people)': 'cost', 'listed_in(type)': 'type', 'listed_in(city)': 'city'})
            #data = data.rename(columns=dict(zip(old_columns_list, new_columns_list))) # renaming the columns
            self.log.info('Renaming columns Completed')
            return data # returning the dataframe after renaming the columns

        except Exception as e:
            self.log.error("Error While Renaming the columns name " + str(e))
            raise e


    def remove_comma(self, data, col):
        """
        Methode: remove_comma
        Description: Custom methode for removing "," or comma form values

        Input: data: pandas dataframe
                  col: column name that needs to be cleaned
        Output: data: pandas dataframe

        On Failure: Raise ValueError, log error in application logger

        Version: 1.0
        """
        try:
            self.log.info('Removing comma from values Started')
            if data[col].dtype != 'object':
                data[col] = data[col].astype(str) # converting the column to string
            data[col] = data[col].str.replace(',', '') # removing the comma from the values
            data[col] = data[col].astype(float) # converting the column to float 
            self.log.info('Removing comma from values Completed')
            return data # returning the dataframe after removing the comma from the values

        except Exception as e:
            self.logger.error("Error While removing comma from columns "+ str(e))
            raise e

    

    # def remove_slash(self, data, col):
    #     """
    #     method: remove_slash
    #     Description: Custome methode for removing "/5".

    #     Input: data: pandas dataframe
    #     Output: data: pandas dataframe

    #     On Failure: Raise ValueError, log error in application logger

    #     Version: 1.0
    #     """
    #     try:
    #         self.log.info('Removing slash from values Started')
    #         data[col] = data[col].astype(str) # converting the column to string
    #         data[col] = data[col].str.replace('/5', '') # removing the slash from the values
    #         data[col] = data[col].astype(float) # converting the column to float 
    #         self.log.info('Removing slash from values Completed')
    #         return data # returning the dataframe after removing the slash from the values

    #     except Exception as e:
    #         self.log.error("Error While removing slash from columns "+ str(e))
    #         raise e
    


    def make_target(self, data, target_col):
        """
        Methode: make_target

        Input: data: (pandas dataframe)
                   : targte_col (str)

        Output: data: (pandas dataframe)

        On Failure: Raise Error, log error in application logger

        Version: 1.0
        """
        try:
            self.log.info('Cleaning Target Columns')
            data[target_col] = data[target_col].astype(str) # converting the column to string
            data = data.loc[data[target_col] != 'NEW'] # removing the rows with NEW]]
            data = data.loc[data[target_col] != "-"]  # removing the "-" from column

            # removing "/5" from rate columns
            data[target_col] = data[target_col].str.replace('/5', '') 
            self.log.info("/5 remove forom rate column")

            # making target columns as float type
            data[target_col] = data[target_col].astype(float) # converting the column to float
            self.log.info('Cleaning Target Columns Completed')

            return data # returning the dataframe after cleaning the target column
        except Exception as e:
            self.log.error("Error While making target "+ str(e))
            raise e 
        


