import os, sys
from sensor import Utils
import pandas as pd
import numpy as np
from sensor.Entity import Config_Entity
from sensor.Entity import Artifact_Entity
from sensor.Exception import SensorException
from sensor.Logger import logging
from sklearn.model_selection import train_test_split



class DataIngestion:
    def __init__(self, data_ingestion_config:Config_Entity.DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(e, sys)
        
    def initiate_data_ingestion(self)->Artifact_Entity.DataIngestionArtifact:
        try:
            # EXPORTING COLLECTION DATA AS DATAFRAME
            logging.info(f"Exporting collection data as pandas dataframe from Database.")
            df:pd.DataFrame = Utils.get_collection_as_dataframe(
                database_name = self.data_ingestion_config.database_name,
                collection_name = self.data_ingestion_config.collection_name)
            

            # REPLACING na VALUES TO NAN IN DATASET
            logging.info(f"Replaing na values to NAN in dataset.")
            df.replace(to_replace= "na", value=np.NAN, inplace= True)
            logging.info(f"Replaced na values to NAN in dataset successfully")

            # CREATING FEATURE STORE FOLDER TO STORING THE DATASET, if not available
            logging.info(f"Creating feature store folder, if not exists.")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok= True)
            logging.info(f"Feature Store Folder is created, if not Exists.")

            # SAVING df TO FEATURE STORE FOLDER
            logging.info(f"Saving Dataframe to feature store folder")
            df.to_csv(path_or_buf= self.data_ingestion_config.feature_store_file_path, index= False, header= True)

            # SPLITTING DATASET TO train AND test SET
            logging.info(f"Splitting dataset to Train and Test")
            train_df, test_df = train_test_split(df, test_size= self.data_ingestion_config.test_size)

            # CREATING DATASET DIRECTORY, if not exists
            logging.info(f"Creating dataset directory, if not exists.")
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok= True)

            # SAVING TRAIN AND TEST DATAFRAME TO FEATURE STORE FOLDER
            logging.info(f"Saving Train Dataset to Feature store folder.")
            train_df.to_csv(path_or_buf= self.data_ingestion_config.train_file_path)

            logging.info(f"Saving Test Dataframe to Feature store folder.")
            test_df.to_csv(path_or_buf= self.data_ingestion_config.test_file_path)

            # PREPARING ARTIFACT
            logging.info(f"Creating FEATURE_STORE_FILE_PATH, TRAIN_FILE_PATH, AND TEST_FILE_PATH variables for Artifact")
            data_ingestion_artifact = Artifact_Entity.DataIngestionArtifact(
                feature_store_file_path = self.data_ingestion_config.feature_store_file_path,
                train_file_path = self.data_ingestion_config.train_file_path,
                test_file_path = self.data_ingestion_config.test_file_path
            )

            logging.info(f"Returning data_ingestion_artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise SensorException(e, sys)
