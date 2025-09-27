from pathlib import Path
import numpy as np
from DrowsinessDetector.data_config.data_cfg import DataTransformationConfig
from DrowsinessDetector.data_config.data_cfg import EvaluationConfig
from DrowsinessDetector.data_config.data_cfg import TrainingConfig
from DrowsinessDetector.components.data_transformation import DataTransformation
from DrowsinessDetector.components.evaluate import ModelEvaluation
from DrowsinessDetector.config_manager.config import ConfigurationManager
from DrowsinessDetector import logger

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
    Make drowsiness predictions on the given video file.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        list: List of predictions for each frame in the video.
    """
    try:
        preprocessed_object = DataTransformation(config=data_transformation_config)
        prediction = list(model_eval.predict_drowsiness(video=Path(video_path), preprocess_obj=preprocessed_object, eval=False))
        return prediction
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return None
