from pathlib import Path
from numpy import isin
import yaml
from ensure import ensure_annotations
from box import ConfigBox
import pickle

from DrowsinessDetector import logger

def read_yaml(yaml_path: Path):
    """
    Reads a YAML file and returns its content as a dictionary.
    
    :param yaml_path: Path to the YAML file.
    :return: Dictionary containing the YAML file content.
    """
    
    try:
        with open(yaml_path, 'r') as file:
            content = ConfigBox(yaml.safe_load(file))
            logger.info("YAML file reading finished.")
        return content
    except Exception as e:
        raise ValueError(f"Error reading the YAML file at {yaml_path}: {e}")
    
    
def create_directories(paths: list):
    """
    Creates directories if they do not exist.    
    :param paths: List of directory paths to create.
    """
    
    for path in paths:
        try:
            if not Path(path).exists():
                logger.info(f"Creating directory: {Path(path)}")
                Path(path).mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {Path(path)}")
            else:
                logger.info(f"Directory already exists: {Path(path)}")
        except Exception as e:
            raise e
        
def save_object(obj_path: Path, obj):
    """
    Saves an object to a file using pickle.
    
    :param obj_path: Path where the object will be saved.
    :param obj: Object to be saved.
    :return: None
    """
    
    try:
        with open(obj_path, 'wb') as file:
            pickle.dump(obj, file)
            logger.info(f"Object saved at {obj_path}")

        
    except Exception as e:
        raise ValueError(f"Error saving the object at {obj_path}: {e}")
    
def load_obj(obj_path: Path):
    """
    Loads an object from a file using pickle.
    
    :param obj_path: Path from where the object will be loaded.
    :return: Loaded object.
    """
    
    try:
        with open(obj_path, 'rb') as file:
            obj = pickle.load(file)
            logger.info(f"Object loaded from {obj_path}")

            # Convert to ConfigBox if it's a dictionary
            if isinstance(obj, dict):
                obj = ConfigBox(obj)

            return obj
    except Exception as e:
        raise ValueError(f"Error loading the object from {obj_path}: {e}")
    

def accuracy_fn(y_preds, y_true) -> float:
    """
    Calculates the accuracy of predictions.
    
    :param y_preds: Predicted values.
    :param y_true: True values.
    :return: Accuracy as a float.
    """
    
    correct = (y_preds == y_true).sum().item()
    total = len(y_true)
    
    accuracy = (correct / total)*100 if total > 0 else 0.0
    logger.info(f"Accuracy calculated: {accuracy}")
    
    return accuracy

    