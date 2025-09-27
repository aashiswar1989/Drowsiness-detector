from DrowsinessDetector.config_manager.config import ConfigurationManager
from DrowsinessDetector.components.train import ModelTraining
from DrowsinessDetector import logger


class TrainingPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config

    def Run_Training(self):
        """
        Starts the training pipeline.
        """
        logger.info("Starting training pipeline.")
        try:
            # Get the training configuration
            training_config = self.config.training_config()
            model_training = ModelTraining(training_config=training_config)
            if training_config.model_name.exists():
                logger.info(f"Model already exists at {training_config.model_name}. Skipping training step.")
                print(f"Model already exists at {training_config.model_name}. Skipping training step.")
                return
            
            model_training.initiate_model_training()
            logger.info("Training pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise e
        
# if __name__ == "__main__":
#     config_manager = ConfigurationManager()
#     training_pipeline = TrainingPipeline(config=config_manager)
#     training_pipeline.Run_Training()