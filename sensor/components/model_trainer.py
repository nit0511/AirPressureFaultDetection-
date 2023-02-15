from sensor.entity import artifact_entity,config_entity
from sensor.exception import SensorException
from sensor.logger import logging
import os, sys
from typing import Optional
from xgboost import XGBClassifier
from sensor import utils
from sklearn.metrics import f1_score


class ModelTrainer:


    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys)

    def fine_tune(self):
        try:
            #write code for Grid Search CV
            pass

        except Exception as e:
            raise SensorException(e, sys)

   

    def train_model(self, x, y):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x,y)
            return xgb_clf
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_trainer(self)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"{type(self.data_transformation_artifact)}")
            logging.info(f"Loading train and test array")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)
            
            logging.info(f"Splitting input and target feature from both train and test arr.")
            x_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            x_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            logging.info(f"train the model")
            model = self.train_model(x=x_train,y = y_train)
            yhat_train = model.predict(x_train)
            logging.info(f"calculating training score")
            f1_train_score = f1_score(y_true = y_train, y_pred = yhat_train)
            yhat_test = model.predict(x_test)
            logging.info(f"calculating testing score")
            f1_test_score = f1_score(y_true = y_test, y_pred = yhat_test)
            logging.info(f"trian score: {f1_train_score} and Test score: {f1_test_score}")
            #check for overfitting or underfitting of expected score
            logging.info(f"checking model if model is underfitting")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected score: {self.model_trainer_config.expected_score}: model actual score is {f1_test_score}")
            logging.info(f"checking if model is overfitting ")
            diff = abs(f1_train_score-f1_test_score)

            if diff>self.model_trainer_config.overfitting_thres:
                raise Exception(f"Train and Test score difference: {diff} is more than\
                 overfitting threshold {self.model_trainer_config.overfitting_thres}")
            logging.info(f"Saving model Object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj = model)

            #prepare artifact
            logging.info(f"preparing the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path= self.model_trainer_config.model_path,
            f1_train_score = f1_train_score, f1_test_score = f1_test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact


        except Exception as e:
            raise SensorException(e, sys)