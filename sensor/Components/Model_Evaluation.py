import os, sys
from sensor.pridictor import ModelResolver
from sensor.Logger import logging
from sensor.Exception import SensorException
from sensor.Entity import Config_Entity, Artifact_Entity

class ModelEvaluation:
    def __init__(self,
                 model_eval_config:Config_Entity.ModelEvaluationConfig,
                 data_ingestion_artifact:Artifact_Entity.DataIngestionArtifact,
                 data_transformation_artifact:Artifact_Entity.DataTransformationArtifact,
                 model_trainer_artifact:Artifact_Entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Model Evaluation {'<<'*20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise SensorException(e, sys)
        

    def initiate_model_evaluation(self)->Artifact_Entity.ModelEvaluationArtifact:
        try:
            # If saved model folder has model we will compare which model is best trained
            # or the model from saved model folder
            
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:
                model_eval_artifact = Artifact_Entity.ModelEvaluationArtifact(is_model_accepted=True,
                                                                              improved_accuracy= None)
                
                logging.info(f"Model Evaluation Artifact: {model_eval_artifact}")
                return model_eval_artifact
        except Exception as e:
            raise SensorException(e, sys)