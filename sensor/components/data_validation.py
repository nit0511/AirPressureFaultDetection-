from sensor.entity import artifact_entity,config_entity
from sensor.exception import SensorException
from sensor.logger import logging
from scipy.stats import ks_2samp
import os, sys
from typing import Optional
from sensor import utils
import numpy as np
import pandas as pd



class DataValidation:

    def __init__(self,
         data_validation_config:config_entity.DataValidationConfig,
         data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data validaion {'<<'*20}")
            self.data_validation_config = data_validation_config
            self. data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise SensorException(e,sys)

 

    def drop_missing_values_columns(self, df:pd.DataFrame,report_key_name:str)->Optional[pd.DataFrame]:
        """ 
        This funtion will drop column which contains missing value more than specified threshold

        df:Accepts a pandas dataframe
        threshold: Percentage criteria to drop a column
        ============================================================================================
        returns Pandas DataFrame if atleast a single column is available after missing columns drop else None
         """
        try:
            threshold = self.data_validation_config.missing_threshold
            logging.info(f"selection column name which contains null abobe to {threshold}")
            
            null_report = df.isna().sum()/df.shape[0]
            #selecting column name which contains null
            drop_column_names = null_report[null_report>threshold].index 
            logging.info(f"Columns to drop: {list(drop_column_names)}")
            self.validation_error[report_key_name] = list(drop_column_names)
            df.drop(list(drop_column_names),axis=1,inplace=True)

            #return None if no column left
            if len(df.columns) == 0:
                return None
            return df
        except Exception as e:
            raise SensorException(e,sys)

    def is_required_columns_exists(self,base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str)->bool:
        try:
            base_colunms = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for base_colunm in base_colunms:
                if base_colunm not in current_columns:
                    logging.info(f"Column:[{base} is not available.]")
                    missing_columns.append(base_colunm)
                
            if len(missing_columns)>0:
                self.validation_error[report_key_name] = missing_columns

                return False
            return True
        except Exception as e:
            raise SensorException(e,sys)
        
    def data_drift(self,base_df:pd.DataFrame, current_df:pd.DataFrame,report_key_name:str):
        try:
            drift_report = dict()
            base_colunms = base_df.columns
            current_columns = current_df.columns
            for base_colunm in base_colunms:
                base_data, current_data = base_df[base_colunm],current_df[base_colunm]
                #Null hypothesis is that both column data drawn form same distribution
                same_distribution = ks_2samp(base_data, current_data)

                if same_distribution.pvalue>0.05:
                    #we are accepting null hypothesis
                    drift_report[base_colunm] = {
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution":True
                    }
                    
                    
                else:
                    drift_report[base_colunm] = {
                        "pvalues": float(same_distribution.pvalue),
                        "same_distribution":False
                    }
            self.validation_error[report_key_name]=drift_report
        except Exception as e:
            raise SensorException(e,sys)
                    
    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"reading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({"na":np.NAN},inplace = True)
            logging.info(f"replace na value in base dataframe")


            base_df = self.drop_missing_values_columns(df=base_df, report_key_name= "missing_value_within_base_dataset")
            logging.info(f"dropping null values columns from base dataframe")

            logging.info(f"reading the train data frame")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)

            logging.info(f"reading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"dropping null values columns from train dataframe")
            train_df = self.drop_missing_values_columns(df = train_df, report_key_name="missing_value_within_train_dataset")
            logging.info(f"dropping null values columns from test dataframe")
            test_df = self.drop_missing_values_columns(df= test_df, report_key_name= "missing_value_within_test_dataset")

            exclude_column = ["class"]
            base_df = utils.convert_columns_to_float(df= base_df, exclude_column = exclude_column)
            train_df = utils.convert_columns_to_float(df= train_df, exclude_column = exclude_column)
            test_df = utils.convert_columns_to_float(df= test_df, exclude_column = exclude_column)

            logging.info(f"is all required columns present train_df")
            tarin_df_columns_status= self.is_required_columns_exists(base_df= base_df, current_df= train_df, report_key_name= "missing_columns_within_train_dataset")
            logging.info(f"is all required columns present in test df")
            test_df_columns_status= self.is_required_columns_exists(base_df= base_df, current_df= test_df ,report_key_name="missing_columns_within_test_dataset")

            if tarin_df_columns_status:
                logging.info(f"As all columns are available in train_df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=train_df,report_key_name="data_drift_within_train_dataset")
            if test_df_columns_status:
                logging.info(f"As all columns are available in train_df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=test_df,report_key_name="data_drift_within_test_dataset")

            #write the report
            logging.info("writing report in Yaml file")

            utils.write_yaml_file(file_path= self.data_validation_config.report_file_path, data= self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            logging.info(f"Data Validation artifact: {data_validation_artifact}")
            return data_validation_artifact


        except Exception as e:
            raise SensorException(e,sys)
