# THIS IS FOR MY CODE INPUTS
import os,sys
from sensor.Exception import SensorException
from sensor.Logger import logging
from datetime import datetime
# from sensor.Config import env_var

# a = env_var.database_name


FILE_NAME = "sensor.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"

class TrainingPipelineConfig:
    def __init__(self):
        logging.info("Running TraningPipelineConfig Function")
        self.artifact_dir = os.path.join(os.getcwd(), "Artifact", f"{datetime.now().strftime('%m%d_%Y__%H_%M_%S')}")

class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            logging.info(f"Running DataIngestionConfig Function")
            self.database_name = "INEURONAPS"
            self.collection_name = "APS-FAULT-DETECTION"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, "feature_store", FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir, "dataset", TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir, "dataset", TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception as e:
            raise SensorException(e, sys)
        
    def to_dict(self)->dict:
        try:
            logging.info(f"Converting DataIngestionConfig Parameters to Dictionary")
            return self.__dict__
        except Exception as e:
            raise SensorException(e, sys)

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
        self.report_file_path = os.path.join(self.data_validation_dir, "report.yaml")
        self.missing_threshold:float = 0.2
        self.base_file_path = os.path.join("aps_failure_training_set1.csv")

class DataTransformationsConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")
        self.transform_object_path = os.path.join(self.data_transformation_dir, "transformer", TRANSFORMER_OBJECT_FILE_NAME)
        self.transformed_train_path = os.path.join(self.data_transformation_dir, "transformed", TRAIN_FILE_NAME.replace("csv", "npz"))
        self.transformed_test_path = os.path.join(self.data_transformation_dir, "transformed", TEST_FILE_NAME.replace("csv", "npz"))
        self.target_encoder_path = os.path.join(self.data_transformation_dir, "target_encoder", TARGET_ENCODER_OBJECT_FILE_NAME)

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
        self.model_path = os.path.join(self.model_trainer_dir, "model", MODEL_FILE_NAME)
        self.expected_score = 0.7
        self.overfitting_threshold = 0.1

        
class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.change_threshold = 0.01



class ModelPusherConfig:...

