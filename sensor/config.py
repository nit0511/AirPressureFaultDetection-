import pymongo
import pandas as pd
import json
from dataclasses import dataclass

#provide the mongodb localhost url to connect python to mongodb.
import os
@dataclass()
class EnvironmentVariable:
    mongo_db_url:str = os.getenv("MONGO_DB_URL")
    aws_access_key_id:str = os.getenv("AWS_ACESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACESS_KEY")
    
    TARGET_COLUMN_MAPPINGS = {
        "pos" :1,
        "neg" :0
}


env_var = EnvironmentVariable()
client = pymongo.MongoClient(env_var.mongo_db_url)


mongo_client = pymongo.MongoClient()

TARGET_COLUMN = "class"