import pymongo
import json
import pandas as pd

DATABASE_NAME = "INEURONAPS"
COLLECTION_NAME = "APS-FAULT-DETECTION"
DATA_FILE_PATH = "aps_failure_training_set1.csv"


client = pymongo.MongoClient("mongodb+srv://INEURON:Saurabh@cluster0.q7kjrda.mongodb.net/?retryWrites=true&w=majority")

# client = pymongo.MongoClient("mongodb+srv://IndustryPWB:Saurabh@cluster0.q7kjrda.mongodb.net/?retryWrites=true&w=majority")

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Columns: {df.shape}")

    # CONVERTING DATAFRAME TO JSON FORMATE TO STORE IN MONGODB
    df.reset_index(drop = True, inplace = True)

    json_record = list(json.loads(df.T.to_json()).values())

    print(json_record[0])

    # UPLOADING DATASET TO MONGODB IN BULK
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

