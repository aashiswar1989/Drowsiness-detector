from DrowsinessDetector.config_manager.config import ConfigurationManager
from DrowsinessDetector.components.data_ingestion import DataIngestion
from DrowsinessDetector import logger


class IngestionPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config

    def Run_Ingestion(self):

        """
        Runs the data ingestion pipeline.
        """
        logger.info("Starting data ingestion pipeline.")
        try:
            # Get the ingestion configuration
            data_ingestion_config = self.config.data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            
            # if data_ingestion_config.target_dir.exists() and data_ingestion_config.test_data.exists():
            #     logger.info(f"Training Data already ingested at {data_ingestion_config.target_dir}. Skipping ingestion step.")                
            #     print(f"Training Data already ingested at {data_ingestion_config.target_dir}. Skipping ingestion step.")
            #     return
            
            data_ingestion.initiate_data_ingestion()
            logger.info("Data ingestion pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Error in ingestion pipeline: {e}")
            raise e
        
if __name__ == "__main__":
    config_manager = ConfigurationManager()
    ingestion_pipeline = IngestionPipeline(config=config_manager)
    ingestion_pipeline.Run_Ingestion()