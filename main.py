from DrowsinessDetector.pipeline.ingestion_pipeline import IngestionPipeline
from DrowsinessDetector.pipeline.transformation_pipeline import TransformationPipeline
from DrowsinessDetector.pipeline.training_pipeline import TrainingPipeline
from DrowsinessDetector.pipeline.evaluation_pipeline import EvaluationPipeline
from DrowsinessDetector.config_manager.config import ConfigurationManager
from DrowsinessDetector import logger

def main():
    """
    Main function to run the Drowsiness Detection pipeline.
    """
    try:
        # Initialize configuration manager
        config = ConfigurationManager()

        # Run data ingestion pipeline
        ingestion_pipeline = IngestionPipeline(config=config)
        ingestion_pipeline.Run_Ingestion()

        # Run data transformation pipeline
        transformation_pipeline = TransformationPipeline(config=config)
        transformation_pipeline.Run_Transformation()

        # Start training pipeline
        training_pipeline = TrainingPipeline(config=config)
        training_pipeline.Run_Training()

        # Start evaluation pipeline
        evaluation_pipeline = EvaluationPipeline(config=config) 
        evaluation_pipeline.Run_Evaluation()

    except Exception as e:
        logger.error(f"An error occurred in the main pipeline: {e}")
        raise e

if __name__ == "__main__":
    main()