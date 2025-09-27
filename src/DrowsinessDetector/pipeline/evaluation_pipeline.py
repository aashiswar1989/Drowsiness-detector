from DrowsinessDetector.components.evaluate import ModelEvaluation
from DrowsinessDetector.config_manager.config import ConfigurationManager
from DrowsinessDetector import logger

class EvaluationPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config

    def Run_Evaluation(self):
        """
        Starts the evaluation pipeline.
        """
        logger.info("Starting evaluation pipeline.")
        try:
            # Get the evaluation configuration
            evaluation_config = self.config.evaluation_config()
            data_transformation_config = self.config.data_transformation_config()
            training_config = self.config.training_config()

            model_evaluation = ModelEvaluation(
                config=evaluation_config,
                data_transformation_config=data_transformation_config,
                training_config=training_config
            )
            model_evaluation.run_evaluation(is_eval=True)
            logger.info("Evaluation pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Error in evaluation pipeline: {e}")
            raise e
        

# if __name__ == "__main__":
#     config = ConfigurationManager()
#     evaluation_pipeline = EvaluationPipeline(config)
#     evaluation_pipeline.Run_Evaluation()
#     logger.info("Evaluation pipeline execution finished.")