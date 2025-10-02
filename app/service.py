from pathlib import Path
import numpy as np
from src.DrowsinessDetector.data_config.data_cfg import DataTransformationConfig
from src.DrowsinessDetector.data_config.data_cfg import EvaluationConfig
from src.DrowsinessDetector.data_config.data_cfg import TrainingConfig
from src.DrowsinessDetector.components.data_transformation import DataTransformation
from src.DrowsinessDetector.components.evaluate import ModelEvaluation
from src.DrowsinessDetector.config_manager.config import ConfigurationManager
from src.DrowsinessDetector import logger

config = ConfigurationManager()
data_transformation_config = config.data_transformation_config()
training_config = config.training_config()
evaluation_config = config.evaluation_config()

model_eval = ModelEvaluation(config=evaluation_config,
                             data_transformation_config=data_transformation_config,
                             training_config=training_config)


MODEL = model_eval.load_model()
model_eval.model = MODEL

def make_predictions(video_path: str):
    """
    Make drowsiness predictions on the uploaded video file.
    
    Args:
        video_path (str): Path to the video file.
        
    Yields:
        dict: Prediction results with confidence and label.
    """
    try:
        preprocessed_object = DataTransformation(config=data_transformation_config)
        for prediction in model_eval.predict_drowsiness(video = Path(video_path), preprocess_obj=preprocessed_object, eval=False):
            logger.info(f'Yielded Prediction for the current sequence is : {prediction}')
            # print(f'Yielded Prediction for the current sequence is : {prediction}')
            yield prediction
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        yield {"error": str(e)}
