from pathlib import Path
from DrowsinessDetector.data_config.data_cfg import DataIngestionConfig
from shutil import copy
import random
from DrowsinessDetector.utils.utils import create_directories
from DrowsinessDetector import logger


class DataIngestion():
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def list_files(self):
        """
        Lists all files in the root directory.
        """
        if not self.config.root_dir.exists():
            logger.error(f"Root directory {self.config.root_dir} does not exist.")
            return []
        
        logger.info(f"Listing files in {self.config.root_dir}.")
        # return [f for f in self.config.root_dir.rglob('*') if f.suffix.lower() in self.config.video_formats]
        awake_dir = self.config.root_dir/'awake'
        drowsy_dir = self.config.root_dir/'drowsy'

        if not awake_dir.is_dir() or not drowsy_dir.is_dir():
            logger.error(f"Expected directories 'awake' and 'drowsy' not found in {self.config.root_dir}.")
            return []
        
        awake_videos = [f for f in awake_dir.glob('*') if f.suffix.lower() in self.config.video_formats]
        drowsy_videos = [f for f in drowsy_dir.glob('*') if f.suffix.lower() in self.config.video_formats]

        return (awake_videos, drowsy_videos)


    def split_files(self, video_files, test_split=0.1):
        """
        Splits the video files into training and testing sets.
        """
        random.seed(42)
        random.shuffle(video_files)
        split_index = int(len(video_files) * (1 - test_split))
        return video_files[:split_index], video_files[split_index:]
    

    def write_split(self, split: str, file_list: list, file_path: Path):
        """
        Writes the split information to a file.
        """
        with open(file_path, 'w') as f:
            for file in file_list:
                f.write(str(file) + '\n')
            logger.info(f"Stored {split} file paths in {file_path}.")


    def copy_files(self, file_list, isTrain=True):
        """
        Copies files from the source directory to the target directory.
        """
        logger.info(f"Copying video files from source directory {self.config.root_dir} to target directory {self.config.target_dir}")

        for file in file_list:
            if isTrain:
                class_name = file.parent.name

                if not(self.config.target_dir/class_name).is_dir():
                    create_directories([self.config.target_dir/class_name])
                    logger.info(f"Created directory for class: {class_name}")

                # Copy the file to the target directory under its class name
                copy(file, self.config.target_dir/class_name)
                logger.info(f'Copied {file.name} to {self.config.target_dir/class_name}.')

            else:
                if not (self.config.test_data).is_dir():
                    logger.error(f"Test data directory: {self.config.test_data} does not exist.")
                copy(file, self.config.test_data)
                logger.info(f'Copied {file.name} to {self.config.test_data}.')
                


    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process by copying training data to the target directory.
        """
        logger.info("Starting data ingestion process.")
        
        try:
            awake_files, drowsy_files = self.list_files()
            if not awake_files and not drowsy_files:
                raise ValueError(f"No training data found in {self.config.root_dir}.")
            
            awake_train, awake_test = self.split_files(awake_files)
            drowsy_train, drowsy_test = self.split_files(drowsy_files)

            train_data = awake_train + drowsy_train
            test_data = awake_test + drowsy_test


            if not(self.config.target_dir.exists() or self.config.test_data.exists()):
                create_directories([self.config.target_dir, self.config.test_data])

            self.write_split('training', train_data, self.config.target_dir/'train_and_val.txt')
            self.write_split('testing', test_data, self.config.test_data/'test.txt')


            # self.copy_files(train_data, isTrain=True)
            # self.copy_files(test_data, isTrain=False)
                
            logger.info(f"Training data copied from {self.config.root_dir} to {self.config.target_dir}.")
            
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise e

