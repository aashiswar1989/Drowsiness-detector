from multiprocessing import process
from pathlib import Path
from urllib import response

from mediapipe.tasks.python.audio import RunningMode
from DrowsinessDetector.data_config.data_cfg import DataTransformationConfig
from DrowsinessDetector.utils.utils import create_directories, save_object
from DrowsinessDetector import logger

import math
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import csv

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

import requests


class DataTransformation():
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    def download_task(self):
        logger.info("Download the MediaPipe Face Landmarker model if not already present.")
        if self.config.landmarker_task.exists():
            logger.info(f"Face landmarker model already exists at {self.config.landmarker_task}.")

        else:
            logger.info(f"Downloading Face Landmakrer task at {self.config.landmarker_task}.")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            response = requests.get(url)
            if response.status_code == 200: 
                with open(self.config.landmarker_task, "wb") as task:
                    task.write(response.content)
                logger.info(f"Face Landmarker task downloaded successfully to {self.config.landmarker_task}.")
            
            else:
                logger.error(f"Failed to download Face Landmarker task. Status code: {response.status_code}")

        


    def initiate_landmarker_task(self):
        """
        Initializes the MediaPipe Face Landmarker task.
        """
                            
        logger.info("Initializing MediaPipe Face Landmarker task.")
        try:
            self.base_options = python.BaseOptions(model_asset_path = self.config.landmarker_task)
            self.options = vision.FaceLandmarkerOptions(
                base_options = self.base_options,
                num_faces = 1,
                running_mode = vision.RunningMode.VIDEO,
                output_face_blendshapes = True
                )
            
            detector = vision.FaceLandmarker.create_from_options(self.options)
            logger.info("Face Landmarker task initialized successfully.")
            return detector

        except Exception as e:
            logger.error(f"Error initializing Face Landmarker task: {e}")
            raise e


    def calculate_EAR(self, eye_coords: dict, eye_lms: list) -> float | None:
        """
        Calculates the Eye Aspect Ratio (EAR) using the coordinates of eye landmarks.
        
        """

        try:
            h_dist = math.dist(eye_coords[eye_lms[0]], eye_coords[eye_lms[1]])
            v1_dist = math.dist(eye_coords[eye_lms[2]], eye_coords[eye_lms[3]])
            v2_dist = math.dist(eye_coords[eye_lms[4]], eye_coords[eye_lms[5]])
            v3_dist = math.dist(eye_coords[eye_lms[6]], eye_coords[eye_lms[7]])
            v4_dist = math.dist(eye_coords[eye_lms[8]], eye_coords[eye_lms[9]])
            v5_dist = math.dist(eye_coords[eye_lms[10]], eye_coords[eye_lms[11]])
        
            v_dist = (v1_dist + v2_dist + v3_dist + v4_dist + v5_dist) / 5
            ear = round((v_dist/h_dist),2)
            return ear
    
        except Exception as e:
            return None

    def calculate_MAR(self, m_coords: dict, m_lms: list) -> float | None:
        """
        Calculates the Mouth Aspect Ratio (MAR) using the coordinates of mouth landmarks.
        """
        try:
            h_dist = math.dist(m_coords[m_lms[0]], m_coords[m_lms[1]])
            v1_dist = math.dist(m_coords[m_lms[2]], m_coords[m_lms[3]])
            v2_dist = math.dist(m_coords[m_lms[4]], m_coords[m_lms[5]])
            v3_dist = math.dist(m_coords[m_lms[6]], m_coords[m_lms[7]])
            v4_dist = math.dist(m_coords[m_lms[8]], m_coords[m_lms[9]])
            v5_dist = math.dist(m_coords[m_lms[10]], m_coords[m_lms[11]])
        
            v_dist = (v1_dist + v2_dist + v3_dist + v4_dist + v5_dist) / 5
            mar = round((v_dist/h_dist),2)
            return mar
    
        except Exception as e:
            return None

    def extract_landmarks(self, video, detector, isTest: bool = False):
        """
        Extracts landmarks from the video using the MediaPipe Face Landmarker task.
        
        """
        # Initialize dictionaries to store coordinates
        le_cord = {}
        re_cord = {}
        m_cord = {}

        train_seq = []
        frame_seq = []

        # Initialize the MediaPipe VideoCapture
        cap = cv2.VideoCapture(video)
        logger.info("Video capture initialized.")

        frame_cnt = 0

        # Process the video frame by frame
        while cap.isOpened():            
            ret, frame = cap.read()
            if not ret:
                break
            frame_cnt += 1
            logger.info(f"Reading frame {frame_cnt} from {video}")

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # Create an Image object from the RGB frame
            mp_img = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame)
        
            # Process the image to extract landmarks
            results = detector.detect_for_video(mp_img, timestamp)

            # Check if landmarks are detected
            if results.face_landmarks:
                # Create a copy of the frame for drawing landmarks
                frame_copy = frame.copy()

                # Extract landmarks for each face
                for faceLM in results.face_landmarks:

                    landmark_list = landmark_pb2.NormalizedLandmarkList()
                    landmark_list.landmark.extend([landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) for landmark in faceLM])

                    
                    # Get coordinates for left eye, right eye, and mouth landmarks and draw them on the frame
                    h,w,c = frame_copy.shape
                    for id, landmarks in enumerate(landmark_list.landmark):
                        cx, cy = int(landmarks.x *w), int(landmarks.y * h)

                        if id in self.config.left_eye_landmarks:
                            le_cord.update({id: (cx, cy)})
                            cv2.circle(frame_copy, (cx, cy), 2, (50, 25, 236), cv2.FILLED)

                        elif id in self.config.right_eye_landmarks:
                            re_cord.update({id: (cx, cy)})
                            cv2.circle(frame_copy, (cx, cy), 2, (50, 25, 236), cv2.FILLED)

                        elif id in self.config.mouth_landmarks:
                            m_cord.update({id: (cx, cy)})
                            cv2.circle(frame_copy, (cx, cy), 2, (50, 25, 236), cv2.FILLED)

                    logger.info("Calculating Eye Aspect Ratio(EAR) for Left eye")
                    EAR_left = self.calculate_EAR(le_cord, self.config.left_eye_landmarks)
                    logger.info(f"Left Eye EAR: {EAR_left}")

                    logger.info("Calculating Eye Aspect Ratio(EAR) for Right eye")
                    EAR_right = self.calculate_EAR(re_cord, self.config.right_eye_landmarks)
                    logger.info(f"Left Eye EAR: {EAR_right}")

                    if EAR_left is None or EAR_right is None:
                        logger.warning("EAR calculation failed for one or both eyes. Skipping this frame.")
                        continue

                    self.EAR_combined = round((EAR_left + EAR_right) / 2, 2)
                    logger.info(f"Combined EAR for left and right eye: {self.EAR_combined}")

                    logger.info("Calculating Mouth Aspect Ratio(MAR)")
                    self.MAR = self.calculate_MAR(m_cord, self.config.mouth_landmarks)
                    logger.info(f"Mouth MAR: {self.MAR}")

                    if self.MAR is None:
                        logger.warning("MAR calculation failed. Skipping this frame.")
                        continue

                    # clear saved dictionaries to be used for next video
                    logger.info("Clearing saved coordinates for next frame.")   
                    le_cord.clear()
                    re_cord.clear()
                    m_cord.clear()

                    frame_seq.append((self.EAR_combined, self.MAR))


                    # Create a sequence of 30 frames for each video
                    if len(frame_seq) == 30:
                        if not isTest:
                            logger.info("Yield a sequence of 30 frames for training/validation data.")                                                                               
                            yield((frame_seq.copy(), video.parent.stem))

                        else:
                            logger.info("Yield a sequence of 30 frames for test(unseen) data.")
                            yield frame_seq.copy()

                        # Creates a rolling sequence of 30 frames for each video . 
                        frame_seq.pop(0)  


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Landmark extraction completed for {video}. Total frames processed: {frame_cnt}") 

        # # Check if train_seq is not empty before returning  
        # logger.info(f"Total training sequences created for {video}: {len(train_seq)}")
        # if not train_seq:
        #     logger.warning("No training data sequences created. Check the video files and landmark extraction.")
        
        # return train_seq

    # Save the  extracted data to csv file
    def save_data(self, train_seq):
        """
        Saves the training sequence data to a CSV file.

        """
        with open(self.config.training_data, 'w') as f:
            writer = csv.writer(f)
            header = [f'EAR_{i}' for i in range(1, len(train_seq[0][0])+1)] + [f'MAR_{i}' for i in range(1, len(train_seq[0][0]) + 1)] + ['Label']
            writer.writerow(header)

            for data, label in train_seq:
                row = [item[0] for item in data] + [item[1] for item in data] + [label]
                writer.writerow(row)



    def split_data(self):
        """
        Splits the training data into training and validation sets.
        """
        logger.info("Splitting data into training and validation sets.")
        try:
            df = pd.read_csv(self.config.training_data)
            X = df.drop(columns=['Label'], axis =1)
            y = df['Label']

            # Split EAR and MAR into separate columns
            X_EAR = X.iloc[:, :30]
            X_MAR = X.iloc[:, 30:]

            # Stacking EAR and MAR so that each frame has both features
            X = np.stack((X_EAR.values, X_MAR.values), axis=2)            

            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            # Split the data into training and validation sets
            logger.info("Performing train-test split.")
            X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y)

            # Standardize the data
            logger.info("Standardizing the training and validation data.")
            X_train_reshaped = X_train.reshape(-1,2)
            X_val_reshaped = X_val.reshape(-1,2)
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train_reshaped) 
            X_val_scaled = scaler.transform(X_val_reshaped)

            # Reshape back to original shape
            X_train = X_train_scaled.reshape(X_train.shape)
            X_val = X_val_scaled.reshape(X_val.shape)

            logger.info("Data splitting completed successfully.")

            # Save the MinMaxScaler object for future use
            save_object(self.config.normalization, scaler)
            logger.info(f"Normalization parameters saved to {self.config.normalization}")


            return {"X_train":X_train, "X_val":X_val, 
                    "y_train": y_train, "y_val":y_val}

        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise e

    def initiate_data_transformation(self): 
        """
        Initiates the data transformation process.
        """

        try:
            # self.frame_seq = []
            # train_seq_video = []
            self.train_seq = []
            
            self.download_task()

            video_file_list = [f for f in self.config.raw_data.rglob("*") if f.suffix.lower() in self.config.video_formats]
            if not video_file_list:
                logger.error(f"No video files found in {self.config.raw_data}.")   

            for video in video_file_list:
                # Extracting EAR and MAR for each video and creating a sequence of 30 frames and storing it in train_seq
                logger.info(f"Processing video: {video}")
                detector = self.initiate_landmarker_task()
                train_seq_video = list(self.extract_landmarks(video, detector))
                self.train_seq.extend(train_seq_video)

            logger.info("Video data stored in sequences of 30 frame per sequence")
            if not self.train_seq:
                logger.error("No training data sequences created. Check the video files and landmark extraction.")
                return
            
            logger.info("Saving training data to CSV file.")
            self.save_data(self.train_seq)
            logger.info(f"Training data saved to {self.config.training_data}")

            logger.info("Splitting the training data into training and validation sets.")
            data_split = self.split_data()
            logger.info("Data transformation completed successfully.")
            save_object(self.config.data_split, data_split)
        
        
        except Exception as e:
            logger.error(f"Error during data transformation: {e}")
            raise e