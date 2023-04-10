import os, sys
from sensor.Logger import logging
from sensor.Exception import SensorException
from sensor.Entity import Config_Entity
from sensor.Components import Data_Ingestion
from sensor.Components.Data_Ingestion import DataIngestion
from sensor.Components.Data_Validation import DataValidation
from sensor.Components.Data_Transformation import DataTransformation
from sensor.Components.Model_Trainer import ModelTrainer
from sensor.Components.Model_Evaluation import ModelEvaluation
from sensor.Utils import get_collection_as_dataframe

print(__name__)
if __name__ == '__main__':
    try:
        
        training_pipeline_config = Config_Entity.TrainingPipelineConfig()
        
        # Data INgestion
        data_ingestion_config = Config_Entity.DataIngestionConfig(training_pipeline_config= training_pipeline_config)

        print(data_ingestion_config.to_dict())

        data_ingestion = DataIngestion(data_ingestion_config= data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        # print(data_ingestion.initiate_data_ingestion())

        #Data Validation
        data_validation_config = Config_Entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config,
                                         data_ingestion_artifact=data_ingestion_artifact)
        
        data_validation_artifact = data_validation.initiate_data_validation()

        # Data Transformation
        logging.info(f"Main Data Transformation Code block Running")
        data_transformation_config = Config_Entity.DataTransformationsConfig(training_pipeline_config=training_pipeline_config)
        logging.info(f"Running Data Transformation")
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                                 data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()


        # Model Trainer
        model_trainer_config = Config_Entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                     data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        # Model Evaluation
        model_eval_config = Config_Entity.ModelEvaluationConfig(training_pipeline_config= training_pipeline_config)
        model_eval = ModelEvaluation(model_eval_config=model_eval_config,
                                     data_ingestion_artifact=data_ingestion_artifact,
                                     data_transformation_artifact=data_transformation_artifact,
                                     model_trainer_artifact=model_trainer_artifact)
        
        model_eval_artifact = model_eval.initiate_model_evaluation()


    except Exception as e:
        print(e)