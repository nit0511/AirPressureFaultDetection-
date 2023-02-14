from sensor.entity import artifact_entity,config_entity
from sensor.exception import SensorException
from sensor.logger import logging
import os, sys
from typing import Optional
from sensor import utils
from sensor.utils import Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import Pipeline
from sklearn.Pipeline import LableEncoder
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sensor.config import TARGET_COLUMN



class DataTrasformation:


    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
        data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)

    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy = "constant", fill_value=0)
            robuar_scaler = RobustScaler()

            constant_pipeline = Pipeline(steps = [
                ('Imputer',simple_imputer),
                ('RobustScaler',robuar_scaler)
            ])
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_transformation(self,)->artifact_entity.DataTransformationArtifact:
        try:
            #reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            #selecting input feature for train and test data frame
            input_feature_train_df = train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN,axis=1)
            #selection target feature for train and test data frame
            target_feature_train_df = train_df(TARGET_COLUMN)
            target_feature_test_df = test_df(TARGET_COLUMN)

            lable_encoder = LableEncoder()
            lable_encoder.fit(target_feature_train_df)
            #transformation on target column
            target_feature_train_arr = lable_encoder.transform(target_feature_train_df)
            target_feature_test_arr = lable_encoder.transform(target_feature_test_df)

            transformation_pipeline = DataTrasformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)
            #transforming input feature
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            smt = SMOTETomek(sampling_strategy = "minority")

            logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target: {target_feature_train_arr.shape}")
            input_feature_train_arr ,target_feature_train_arr = smt.fit_resample(input_feature_train_arr ,target_feature_train_arr)
            logging.info(f"after resampling in training set Input: {input_feature_train_arr.shape} Target: {target_feature_train_arr.shape}")

            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target: {target_feature_test_arr}")
            input_feature_test_arr ,target_feature_test_arr = smt.fit_resample(input_feature_test_arr ,target_feature_test_arr)
            logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target: {target_feature_test_arr}")

            #target encoder
            train_arr = np.c(input_feature_train_arr ,target_feature_train_arr)
            test_arr = np.c(input_feature_test_arr ,target_feature_test_arr)

            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path, array= train_arr)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path, array= test_arr)

            utils.save_object(file_path=self.data_transformation_config.transfor_object_path, obj= transformation_pipeline)
            utils.save_object(file_path=self.data_transformation_config.target_encoder_path, obj= lable_encoder)

            data_transformaton_artifact = artifact_entity.DataTransformationArtifact(
                transfor_object_path = self.data_transformation_config.transfor_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path
            )

            logging.info(f"Data transformation object{data_transformation_artifact}")

        except Exception as e:
            raise SensorException(e, sys)
        
        

