from pathlib import Path

from DrowsinessDetector.constants import *
from DrowsinessDetector.utils.utils import read_yaml, create_directories
from DrowsinessDetector.data_config.data_cfg import DataIngestionConfig
from DrowsinessDetector.data_config.data_cfg import DataTransformationConfig
from DrowsinessDetector.data_config.data_cfg import TrainingConfig
from DrowsinessDetector.data_config.data_cfg import EvaluationConfig
from DrowsinessDetector import logger


class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        self.config_path = config_filepath
        self.params_path = params_filepath

        self.config = read_yaml(self.config_path)
        self.params = read_yaml(self.params_path)

        create_directories([self.config.artifacts_root])
        
    def data_ingestion_config(self)-> DataIngestionConfig:
        """
        Returns Data Ingestion configuration as a dataclass.
        """
        logger.info("Creating data ingestion configuration.")
        create_directories([self.config.data_ingestion.target_dir, self.config.data_ingestion.test_data])
        data_ingestion_config = DataIngestionConfig(
            root_dir = Path(self.config.data_ingestion.root_dir),
            target_dir = Path(self.config.data_ingestion.target_dir),
            test_data = Path(self.config.data_ingestion.test_data),
            video_formats = self.params.VIDEO_FORMATS
        )


        return data_ingestion_config
    
    def data_transformation_config(self)-> DataTransformationConfig:
        """
        Returns Data Transformation configuration as a dataclass.
        """
        logger.info("Creating data transformation configuration.")
        create_directories([self.config.data_transformation.root_dir])

        raw_data = Path(self.config.data_ingestion.target_dir)

        data_transform_config = DataTransformationConfig(
            root_dir = Path(self.config.data_transformation.root_dir),
            raw_data = raw_data,
            training_data = Path(self.config.data_transformation.training_data),
            data_split = Path(self.config.data_transformation.data_split),
            normalization = Path(self.config.data_transformation.normalization),
            landmarker_task = Path(self.config.data_transformation.landmarker_task),
            left_eye_landmarks= self.params.LEFT_EYE,
            right_eye_landmarks= self.params.RIGHT_EYE,
            mouth_landmarks= self.params.MOUTH,
            video_formats = self.params.VIDEO_FORMATS
        )

        return data_transform_config
    

    def training_config(self) -> TrainingConfig:
        """
        Returns Training configuration as a dataclass.
        """
        logger.info("Creating training configuration.")
        create_directories([self.config.training.model_dir])

        data = Path(self.config.data_transformation.data_split)

        training_config = TrainingConfig(
            model_dir = Path(self.config.training.model_dir),
            data = data,
            model_name = Path(self.config.training.model_name),
            learning_rate = self.params.LEARNING_RATE,
            epochs = self.params.EPOCHS,
            batch_size = self.params.BATCH_SIZE,
            no_features = self.params.NUM_FEATURES,
            no_lstm_units = self.params.NUM_LSTM_UNITS,
            no_lstm_layers = self.params.NUM_LSTM_LAYERS,
            dropout = self.params.DROPOUT
        )

        return training_config
    
    def evaluation_config(self) -> EvaluationConfig:
        """
        Returns the Validation Configuration as a dataclass.
        """
        logger.info("Creating validation configuration.")

        evalation_config = EvaluationConfig(
            model_path = Path(self.config.training.model_name),
            test_data = Path(self.config.data_ingestion.test_data),
            video_formats = self.params.VIDEO_FORMATS
        )

        return evalation_config