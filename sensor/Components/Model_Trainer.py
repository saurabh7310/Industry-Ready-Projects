from sensor.Entity import Artifact_Entity, Config_Entity
from sensor.Exception import SensorException
from sensor.Logger import logging
from typing import Optional
import os,sys 
from xgboost import XGBClassifier
from sensor import Utils
from sklearn.metrics import f1_score






class ModelTrainer:

    def __init__(self, model_trainer_config:Config_Entity.ModelTrainerConfig,
                 data_transformation_artifact:Artifact_Entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Training {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)
        
    def train_model(self, x, y):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x, y)
            return xgb_clf
        except Exception as e:
            raise SensorException(e, sys)
        
    def fine_tune(self):
        try:
            # Write code for Grid Search CV
            pass
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_trainer(self,)->Artifact_Entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading Train and Test Array")
            train_arr = Utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = Utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting Target and Input Feature from both train and test arr")
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info(f"Train the Model")
            model = self.train_model(x = x_train, y= y_train)

            logging.info(f"Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true= y_train, y_pred=yhat_train)

            logging.info(f"Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred= yhat_test)

            logging.info(f"train score: {f1_train_score} and test score: {f1_test_score}")
            # Checking for Overfitting or Underfitting or Expected Score
            logging.info(f"Checking if our model is Underfitting or not")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception("Model is not good as it is not able to give expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {f1_test_score}")
            
            logging.info(f"Checking if our model is Overfitting or not")
            diff = abs(f1_train_score-f1_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and Test score diff: {diff} is more than Overfitting Threshold {self.model_trainer_config.overfitting_threshold}")
            
            # Saved the Trained Model
            Utils.save_object(file_path=self.model_trainer_config.model_path, obj= model)

            # Prepare Artifact
            model_trainer_artifact = Artifact_Entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path,
                                                                         f1_train_score=f1_train_score,
                                                                         f1_test_score=f1_test_score)
            
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")

            return model_trainer_artifact
        except Exception as e:  
            raise SensorException(e, sys)
