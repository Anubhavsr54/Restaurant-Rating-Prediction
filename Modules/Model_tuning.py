from Modules.setup_logger import setup_logger
from Modules.Splitting_Scaling import *
from sklearn.ensemble import RandomForestRegressor

from math import e
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import bz2
import pickle
#import _pickle as cPickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.ensemble import ExtraTreesRegressor
from Modules.setup_logger import setup_logger
from Modules.scores_and_errors import Score

from compress_pickle import dump

class Parameter_tuning:
    def __init__(self, df):
        self.df = df
        self.folder = './logs/'
        self.filename = 'Model_tuning.txt'
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = setup_logger("Model_tuning", self.folder+self.filename)
        
        self.log_object.info('Started calling Splitting_Scaling file')
        
        self.split_obj=Splitting_And_Scaling(df)
        
        self.dict={}
        self.log_object.info('Splitting_Scaling file called Successfully.')
        
        self.obj_score=Score()
        
        

    def parameters(self):
        """
        Method: parameters
        Description: This method is used to define the parameters for the model
        Parameters: None
        Return: parameters for individual models

        Version: 1.0
        """
        self.log_object.info('Trying to set hyper-parameters')
        et_parameters={'n_estimators': [100,200,300,400,500,600],
                       'max_features': ['auto', 'sqrt'],
                       'max_depth': [int(x) for x in np.linspace(5, 30, num = 6)],
                       'min_samples_split': [2, 5, 10],
                       'min_samples_leaf': [1, 2, 5]
                      }
        
        rf_parameters={'n_estimators': [100,200,300,400,500,600],
                       'max_features': ['auto', 'sqrt'],
                       'max_depth': [int(x) for x in np.linspace(5, 30, num = 6)],
                       'min_samples_split': [2, 5,10],
                       'min_samples_leaf': [1, 2,4],
                       'bootstrap': [True, False]
                      }
        self.log_object.info('Hyper-parameters is successfully set.')

        return et_parameters, rf_parameters
        
    def et_tuning(self):
        """
        Method: et_tuning
        Description: This method is used to tune the parameters for the Extra Trees Regressor model
        Parameters: None
        Return: Best hyperparameters for the Extra Trees Regressor model and tuned model

        Version: 1.0
        """
        self.log_object.info('Train-Test Split') 
        x_train,x_test,y_train,y_test=self.split_obj.scaling()
        
        try:
            self.log_object.info('ExtraTree Regressor: Model Tuning Started') 
            et_parameters = self.parameters()[0]
            et_reg = ExtraTreesRegressor()
            random_et = RandomizedSearchCV(estimator=et_reg,
                                        param_distributions=et_parameters,
                                        cv=5,
                                        scoring='neg_root_mean_squared_error',
                                        n_iter=10,
                                        n_jobs=-1,
                                        verbose=0,
                                        random_state=42
                                        )
            
            random_et.fit(x_train,y_train) 
            best_param = random_et.best_params_
            self.log_object.info('ExtraTree Regressor:Best Parameters found.')


            et_model = ExtraTreesRegressor(n_estimators = best_param['n_estimators'],
                                            max_features = best_param['max_features'],
                                            max_depth = best_param['max_depth'],
                                            min_samples_split = best_param['min_samples_split'],
                                            min_samples_leaf = best_param['min_samples_leaf'])
            
            self.log_object.info('ExtraTree Regressor:Using best parameter model tuning done.')
            
            et_model.fit(x_train,y_train)
            self.log_object.info('ExtraTree Regressor:Training data fitted to tuned model.') 
        except Exception as e:
            self.log_object.info('Error in ExtraTree Regressor Tuning.'+str(e)) 


        try:
            self.log_object.info('ExtraTree Regressor:Finding train and test accuracy')
            
            #train score and y_predict
            et_train_score = et_model.score(x_train,y_train)
            y_pred = et_model.predict(x_test)
            
            #train score
            et_test_score = self.obj_score.evaluation_r2_score(y_test,y_pred)           
            self.log_object.info(f'ExtraTree Regressor:train accuracy:{et_train_score} and Test accuracy: {et_test_score}')
            
            #Mean absolute error
            et_mae=self.obj_score.mae(y_test,y_pred)
            self.log_object.info(f"Mean Absolute Error(mae) on Testing Data: {str(et_mae)}")
            
            #Root mean square
            et_rmse=self.obj_score.rmse(y_test,y_pred)
            self.log_object.info(f"Root Mean Squared Error(rsme) on Testing Data: {str(et_rmse)}")

            #Saving into dictionary
            self.log_object.info('ExtraTree Regressor:Saving accuracy scores and error metrices into dictionary')
            self.dict['ExtratreeRegressor']=[et_model, et_train_score, et_test_score, et_mae, et_rmse]
            
            self.log_object.info('ExtraTree Regressor:All the details are Succesfully saved into dictionary')
            
        except Exception as e:
            self.log_object.info('Error in ExtraTree Regressor: '+str(e)) 
                
                
    def rf_tuning(self):
        """
        Method: rf_tuning
        Description: This method is used to tune the parameters for the Random forest model
        Parameters: None
        Return: Best hyper-parameters for the Random forest model and tuned model

        Version: 1.0
        """
        try:
            rf_parameters = self.parameters()[1]
            x_train, x_test, y_train, y_test = self.split_obj.scaling()
            self.log_object.info('RandomForestRegressor: Train-Test Split') 
            
            rf_reg = RandomForestRegressor()
            self.log_object.info('RandomForestRegressor: Model Tuning Started') 
            
            random_rf = RandomizedSearchCV(estimator=rf_reg,
                                        param_distributions=rf_parameters,
                                        cv=5,
                                        scoring='neg_root_mean_squared_error',
                                        n_iter=10,
                                        n_jobs=-1,
                                        verbose=0,
                                        random_state=45
                                        )
            random_rf.fit(x_train,y_train)
            best_param = random_rf.best_params_
            self.log_object.info('RandomForestRegressor: Best parameter found.') 
    
            rf_model = RandomForestRegressor(n_estimators = best_param['n_estimators'],
                                            min_samples_split = best_param['min_samples_split'],
                                            min_samples_leaf = best_param['min_samples_leaf'],
                                            max_features = best_param['max_features'],
                                            max_depth = best_param['max_depth'],
                                            bootstrap=best_param['bootstrap'])
          
            rf_model.fit(x_train,y_train)
            self.log_object.info('RandomForestRegressor: Success-fit model using best params') 
            
          
            self.log_object.info('RandomForestRegressor:Finding train and test accuracy')
            #train score and y_predict
            rf_train_score = rf_model.score(x_train,y_train)
            y_pred = rf_model.predict(x_test)
            
            #train score
            rf_test_score = self.obj_score.evaluation_r2_score(y_test,y_pred)           
            self.log_object.info(f'RandomForest Regressor:train accuracy:{rf_train_score} and Test accuracy: {rf_test_score}')
            
            #Mean absolute error
            rf_mae=self.obj_score.mae(y_test,y_pred)
            self.log_object.info(f"Mean Absolute Error(mae) on Testing Data: {str(rf_mae)}")
            
            #Root mean square
            rf_rmse=self.obj_score.rmse(y_test,y_pred)
            self.log_object.info(f"Root Mean Squared Error(rsme) on Testing Data: {str(rf_rmse)}")

            #Saving into dictionary
            self.log_object.info('RandomForest Regressor:Saving accuracy scores and error metrices into dictionary')
            self.dict['RandomForest Regressor']=[rf_model, rf_train_score, rf_test_score, rf_mae, rf_rmse]
            
            self.log_object.info('RandomForest Regressor:All the details are Succesfully saved into dictionary')

        except Exception as e:
            self.log_object.info('Error in RandomForest Regressor: '+str(e))


    def algo_run(self):
        self.et_tuning()
        self.rf_tuning()
        
    def model_result(self):
        """
        Method: model_result
        Description: This method is used to print the best model and the corresponding score
        Parameters: None
        Return: Store the scores obtain from different algorithms in a dictionary

        Version: 1.0
        """
        try:
            self.log_object.info('Finding best model..')
            self.algo_run()
            d = self.dict
            d = sorted(d.items(), key=lambda a:a[1][1])
                        
            best_model_name = d[0][0]
            best_model_object = d[0][1][0]
            #best_model_object1 = d[0][1][0]
            best_model_test_score = d[0][1][2]
            best_model_rmse=d[0][1][-1]           
            
            if not os.path.isdir('./bestmodel/'):
                os.mkdir('./bestmodel/')
            
            #with open('./bestmodel/'+best_model_name+'.pkl','wb') as file:
                #pickle.dump(best_model_object,file)
                #dump(best_model_object, file, compression=None,set_default_compression=False)

            sfile = bz2.BZ2File('./bestmodel/'+best_model_name+'.pkl', 'w')
            pickle.dump(best_model_object, sfile)

           # with bz2.BZ2File('./bestmodel1/' + ‘.pbz2’,‘w’) as file:
           #     cPickle.dump(best_model_object1, file)

            #    best_model_object1 = bz2.BZ2File(file, ‘rb’)
             #   best_model_object1.pbz2 = cPickle.load(best_model_object1)

            return best_model_name, best_model_test_score, best_model_rmse
        except Exception as e:
            self.log_object.info('Error in finding best model'+str(e))