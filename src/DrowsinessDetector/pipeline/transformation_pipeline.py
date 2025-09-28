from DrowsinessDetector.config_manager.config import ConfigurationManager
from DrowsinessDetector.components.data_transformation import DataTransformation
from DrowsinessDetector import logger

class TransformationPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config

    def Run_Transformation(self):
        """
        Runs the data transformation pipeline.
        """
        logger.info("Starting data transformation pipeline.")
        try:
            # Get the transformation configuration
            data_transformation_config = self.config.data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            # if data_transformation_config.training_data.exists():
            #     logger.info("Training data already exists. Skipping data transformation.")
            #     print("Training data already exists. Skipping data transformation.")
            #     return
            
            data_transformation.initiate_data_transformation()
            logger.info("Data transformation pipeline completed successfully.")

            
        except Exception as e:
            logger.error(f"Error in transformation pipeline: {e}")
            raise e
        
if __name__ == "__main__":
    config_manager = ConfigurationManager()
    transformation_pipeline = TransformationPipeline(config=config_manager)
    transformation_pipeline.Run_Transformation()