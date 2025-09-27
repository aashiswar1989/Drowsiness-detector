from dataclasses import dataclass
from pathlib import  Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    target_dir: Path
    test_data: Path
    video_formats: list
    

@dataclass
class DataTransformationConfig:
    root_dir: Path
    raw_data: Path
    training_data: Path
    data_split: Path
    normalization: Path
    landmarker_task: Path
    left_eye_landmarks: list
    right_eye_landmarks: list
    mouth_landmarks: list
    video_formats: list

@dataclass
class TrainingConfig:
    model_dir: Path
    model_name: Path
    data: Path 
    learning_rate: float
    epochs: int
    batch_size: int
    no_features: int
    no_lstm_units: int
    no_lstm_layers: int
    dropout: float

@dataclass
class EvaluationConfig:
    model_path: Path
    test_data: Path
    video_formats: list
