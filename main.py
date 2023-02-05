from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils import get_collection_as_dataframe
import sys, os
from sensor.entity import config_entity
if __name__ == '__main__':
     try:
          trainning_pipeline_config = config_entity.TrainnigPipelineConfig()
          data_ingestion_config = config_entity.DataIngestionConfig(trainning_pipeline_config=trainning_pipeline_config)
          print(data_ingestion_config.to_dict())
     except Exception as e:
            raise SensorException(e, sys)