from pathlib import Path
import numpy as np
from DrowsinessDetector.utils.utils import load_obj, accuracy_fn
from DrowsinessDetector.data_config.data_cfg import EvaluationConfig
from DrowsinessDetector.data_config.data_cfg import DataTransformationConfig
from DrowsinessDetector.data_config.data_cfg import TrainingConfig
from DrowsinessDetector.components.data_transformation import DataTransformation
from DrowsinessDetector.components.train import DrowsinessDetectorModel, ModelTraining

import torch
from torch import nn

from DrowsinessDetector import logger

class ModelEvaluation:
    def __init__(self, config: EvaluationConfig, 
                 data_transformation_config: DataTransformationConfig,
                 training_config: TrainingConfig):
        self.config = config
        self.data_transformation_config = data_transformation_config
        self.training_config = training_config


    def list_test_videos(self, test_video_path: Path):
        """
        Lists all the test video files in the specified directory.
        """
        if not test_video_path.exists():
            logger.error(f"Test video path {test_video_path} does not exist.")
            return []
        
        logger.info(f"Loading video from {test_video_path}")
        with open(test_video_path, 'r') as f:
            video_files = [Path(line.strip()) for line in f.readline() if line.strip()] 
        
        # video_files = [f for f in test_video_path.rglob('*') if f.suffix.lower() in self.config.video_formats]

        if not video_files:
            logger.warning(f"No video files found in {test_video_path}.")
            return []
        
        logger.info(f"Found {len(video_files)} video files for evaluation.")
        # print(f"Found {len(video_files)} video files for evaluation.")
        return video_files
    
    def predict_drowsiness(self, video, preprocess_obj: DataTransformation, eval = True):
        """
        Prepares the test data for evaluation.
        
        """
        
        # Initialize the face landmarker task
        detector = preprocess_obj.initiate_landmarker_task()
        if not detector:
            # print("Detector not initialized.")
            logger.error('Error initializing the face landmarker task.')
            return None

        logger.info(f"Preparing data for evaluation from video: {video}")

        y_test_true = []
        y_test_pred = []

        true_label = None
        if eval:
            if "drowsy" in video.stem.lower():
                true_label = 1
            elif "awake" in video.stem.lower():
                true_label = 0
            else:
                logger.warning("No label found. Switching to inference mode.")
                eval = False


        # Load the normalization scaler
        scaler = load_obj(self.data_transformation_config.normalization)

        # test_seq_cnt = 1
        # Extract landmarks and perform preprocessing and feed to model for prediction for every 30 frames in the video
        for test_seq in preprocess_obj.extract_landmarks(video, detector, isTest=True):
            if not test_seq:
                logger.error(f"Error extracting landmarks from video: {video}")
                continue
            # test_seq_cnt += 1 
            test_seq = np.array(test_seq)
            logger.info(f"Extracted sequence shape: {test_seq.shape}")
            # Normalize the test sequence
            logger.info("Normalizing the test sequence.")
            
            test_seq_scaled = scaler.transform(test_seq)
            test_seq = test_seq_scaled.reshape(1, 30, 2)  # Reshape to match model input

            logger.info(f"Test sequence prepared with shape: {test_seq.shape}")
            pred_prob, pred_label = self.evaluate_model(self.model, test_seq)
            logger.info(f"Confidence: {pred_prob:.2f}, Predicted label: {pred_label}")

            if eval:
                y_test_true.append(true_label)        
                y_test_pred.append(pred_label)
            else:
                # print(f'Test sequences processed: {test_seq_cnt}', end='\r')
                yield {
                    "confidence": round(pred_prob,2),
                    "predicted_label": "Drowsy" if int(pred_label) == 1 else "Awake"
                }

        if eval and len(y_test_true) > 0:   
            test_acc = accuracy_fn(np.array(y_test_pred), np.array(y_test_true))
            logger.info(f"Current Test Accuracy: {test_acc:.2f}%")

            return {
                "Test Accuracy": test_acc
            }


    def load_model(self):
        """
        Loads the model stored during training.
        """
        model = DrowsinessDetectorModel(self.training_config)
        model.create_model()
        model.load_state_dict(torch.load(self.config.model_path, map_location=torch.device("cpu")))
        model.to(ModelTraining.select_device())

        logger.info(f"Model loaded from {self.config.model_path}")
        return model

    def evaluate_model(self, model, sequence):
        """
        Evaluates the model on the provided sequence.
        
        """
        with torch.inference_mode():
            model.eval()            
            # Ensure the sequence is a tensor and move it to the correct device
            sequence = torch.tensor(sequence, dtype = torch.float32).to(ModelTraining.select_device())
            logger.info("Running inference on the model.")
            # Get the model prediction
            prediction = model(sequence).squeeze(dim = 1)
            logger.info(f"Model prediction shape: {prediction.shape}")
            # Convert prediction to binary label
            logger.info("Converting prediction to binary label.")
            pred_prob = torch.sigmoid(prediction)
            pred_label = torch.round(pred_prob)
            logger.info(f"Prediction for the sequence: {pred_label.item()} with probability {pred_prob.item():.4f}")

        return (pred_prob.item(), pred_label.item())


    def run_evaluation(self, is_eval = True):
        """
        Runs the evaluation process.
        
        """
        logger.info("Starting model evaluation.")

        # Load the model
        self.model = self.load_model()        
        if not self.model:
            logger.error("Model loading failed. Exiting evaluation.")
            return
        
        test_videos = self.list_test_videos(self.config.test_data/'test.txt')
        if not test_videos:
            logger.error("No test videos found. Exiting evaluation.")
            return
        
        # Initialize the data transformation object
        preprocess_obj = DataTransformation(self.data_transformation_config)
        if not preprocess_obj:
            logger.error("Error initializing data transformation object.")
            return
        

        for video in test_videos:
            # Process each video before evaluation
            pred = self.predict_drowsiness(video, preprocess_obj, eval=is_eval)
            logger.info(f"Evaluation completed for video: {video}")
            print(f"Evaluation completed for video: {video}")