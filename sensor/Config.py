import os
import json
import pymongo
import pandas as pd
from dataclasses import dataclass
from sensor.Logger import logging

# PROVIDING MONGO_DB_URL TO CONNECT python TO mongodb
@dataclass
class EnvironmentVariable:
    mongo_db_url:str = os.getenv("MONGO_DB_URL")
    aws_access_key_id:str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key:str = os.getenv("AWS_SECRET_ACCESS_KEY")
    database_name:str = os.getenv("DATABASE_NAME")

TARGET_COLUMNS_MAPPING = {
    "pos":1,
    "neg":0
}

env_var = EnvironmentVariable()

mongo_client = pymongo.MongoClient(env_var.mongo_db_url)
TARGET_COLUMN = "class"
