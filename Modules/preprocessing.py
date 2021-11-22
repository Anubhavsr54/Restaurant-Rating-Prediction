from Modules.setup_logger import setup_logger
from Modules.data_loader import DataGetter
from Modules.preprocessor import Preprocessor
from Modules.data_transformation import DataTransformation

# creating custome class to process the data and saved it to loacl for feed to model.


class Preprocessing():
    """
    Write docstring here
    """

    def __init__(self):
        # create instance of logger
        self.log = setup_logger('preprocessing', "logs/preprocessing.log")
        self.data_loader = DataGetter()  # create instance of data loader
        self.preprocessor = Preprocessor()  # create instance of preprocessor
        # create instance of data transformation
        self.data_transformation = DataTransformation()

    def preprocess_data(self, data_path):
        """
        Method: preprocess_data.
        Description: This methode is for preprocessing the data.
        """
        try:
            self.log.info("Start preprocessing data")

            # Step 1
            # get the data from the data loader
            self.log.info("Loading data... Started")
            df = self.data_loader.data_getter(data_path)  # load data as dataframe
            self.log.info("Data loaded... Completed")

            # Step 2
            # renaming columns name
            self.log.info("Renaming Column name.. Started")

            df = self.data_transformation.rename_columns(data=df)
            self.log.info("Renaming COlumn name.. Compleated")

            # Step 3
            # droping the unwanted columns, came from EDA
            self.log.info("Dropping unwanted columns... Started")
            df = self.preprocessor.drop_columns(data=df, col_list=[
                                                                'dish_liked',
                                                                'cuisines',
                                                                'city'  
                                                            ])
            self.log.info("Unwanted columns dropped... Completed")

            # Step 4
            # droping null values from the dataframe
            self.log.info("Dropping null values... Started")
            df = self.preprocessor.drop_null(data=df)
            self.log.info("Null values dropped... Completed")

            # Step 5
            # droping duplicate values from the dataframe
            self.log.info("Dropping duplicate values... Started")
            df = self.preprocessor.drop_duplicates(data=df)
            self.log.info("Duplicate values dropped... COmpeted")

            # steps 6 
            # remove comma form cost column
            self.log.info("Removing comma from cost column... Started")
            df = self.data_transformation.remove_comma(data=df, col="cost") 
            self.log.info("Removing comma from cost column... Completed")

            # Step 7
            # maek target column
            df = self.data_transformation.make_target(data=df, target_col="rate")
            self.log.info("Target column created... Completed")

            # Step 8
            # Encoding categorical columns
            self.log.info("Encoding categorical columns... Started")
            # Encoding "online order"
            df, online_order_dict = self.preprocessor.label_encoding(data=df, cat_col="online_order")
            # saving in json file
            self.data_loader.save_json(online_order_dict, "./encoding_dict/online_order.json")

            # Encoding "book_table"
            df, book_table_dict = self.preprocessor.label_encoding(data=df, cat_col="book_table")
            # saving in json file
            self.data_loader.save_json(book_table_dict, "./encoding_dict/book_table.json")

            # Encoding location
            df, location_dict = self.preprocessor.label_encoding(data=df, cat_col="location")
            # saving in json file
            self.data_loader.save_json(location_dict, "./encoding_dict/location.json")

            # Encoding rest_type
            df, rest_type_dict = self.preprocessor.label_encoding(data=df, cat_col="rest_type")
            # saving in json file
            self.data_loader.save_json(rest_type_dict, "./encoding_dict/rest_type.json")

            # Encoding type
            df, type_dict = self.preprocessor.label_encoding(data=df, cat_col="type")
            # saving in json file
            self.data_loader.save_json(type_dict, "./encoding_dict/type.json")
            self.log.info("Encoding categorical columns... Completed")

            # Step 7
            # saving the dataframe as csv file for model feeding
            self.log.info("Saving dataframe as csv file... Started")
            # self.data_loader.data_saved(df, "./data/preprocessed_data.csv")
            self.log.info("Saving dataframe as csv file... Completed")
            return df # return the dataframe


        except Exception as e:
            self.log.info("Error on Preprocessing data " + str(e))