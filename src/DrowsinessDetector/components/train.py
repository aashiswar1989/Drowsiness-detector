from pathlib import Path
from DrowsinessDetector.data_config.data_cfg import TrainingConfig
from DrowsinessDetector.utils.utils import create_directories, load_obj, accuracy_fn, save_object
from DrowsinessDetector import logger

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ModelTraining():
    def __init__(self, training_config: TrainingConfig):
        self.config = training_config
        create_directories(([self.config.model_dir]))

    def loss_function(self):
        """
        Returns the loss function for the model.
        
        """
        return nn.BCEWithLogitsLoss()

    def optimizer(self, model):
        """
        Returns an optimizer for the model.

        """
        params = model.parameters()
        return torch.optim.Adam(params = params, lr = self.config.learning_rate)

    
    @staticmethod
    def select_device():
        """
        Selects the device for training (GPU or CPU).
        
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using GPU for training.")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for training.")
        
        return device
    
    def save_model(self, model: nn.Module):
        """
        Saves the trained model to the specified path.
        
        """
        logger.info(f"Saving the model to {self.config.model_name}")
        torch.save(model.state_dict(), self.config.model_name)
        logger.info(f"Model saved successfully at {self.config.model_name}")


    def train_and_val_loop(self, model: nn.Module, data):
        """
        Training loop for the  model
        
        """

        for epoch in range(self.config.epochs):

            # Training step
            logger.info(f"Starting training for epoch {epoch + 1}")
            model.train()

            train_loss, train_accuracy = 0, 0

            for batch, (data.X_train, data.y_train) in enumerate(self.train_dataloader):
                logger.info(f"Training batch {batch+1} of {len(self.train_dataloader)} batches")

                # Move data to the selected device
                X_train, y_train = data.X_train.to(self.select_device()), data.y_train.to(self.select_device())

                # Forward pass
                y_preds = model(X_train).squeeze(dim = 1)
                
                # Calculate loss and accuracy
                loss = self.loss_function()(y_preds, y_train)

                train_loss += loss.item()
                y_preds_labels = torch.round(torch.sigmoid(y_preds))
                train_accuracy += accuracy_fn(y_preds_labels, y_train)

                # Backward pass and optimization
                self.optimizer(model).zero_grad()
                loss.backward()
                self.optimizer(model).step()

            train_loss /= len(self.train_dataloader)
            train_accuracy /= len(self.train_dataloader)
            logger.info(f"Training for Epoch {epoch+1} completed.")
            logger.info("*"*50)

            # Validation step            
            logger.info(f"Starting validation for epoch {epoch+1}")
            model.eval()
            val_loss, val_accuracy = 0,0

            with torch.inference_mode():
                for batch, (data.X_val, data.y_val) in enumerate (self.test_dataloader):
                    logger.info(f"Validation Batch {batch+1} of {len(self.test_dataloader)} batches")

                    # Move data to the selected device
                    X_val, y_val = data.X_val.to(self.select_device()), data.y_val.to(self.select_device())

                    # Forward pass
                    y_val_preds = model(X_val).squeeze(dim = 1)

                    # Calculate loss and accuracy
                    val_loss += self.loss_function()(y_val_preds, y_val).item()
                    y_val_labels = torch.round(torch.sigmoid(y_val_preds))
                    val_accuracy += accuracy_fn(y_val_labels, y_val)

                # Average the loss and accuracy
                val_loss /= len(self.test_dataloader)
                val_accuracy /= len(self.test_dataloader)
                logger.info(f"Validation for Epoch {epoch+1} completed.")
                logger.info("*"*50)

            logger.info(f"Epoch {epoch+1} completed. Training Loss: {train_loss}, Training Accuracy: {train_accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
            logger.info('\n\n')            

        logger.info(f"Model training completed successfully for {self.config.epochs} epochs.")


    def initiate_model_training(self):
        """
        Initiates the model training process.
        
        """
        logger.info("Starting model training process.")

        try:
            #Load the training data
            logger.info("Loading the data for training process.")
            data = load_obj(self.config.data)
            logger.info("Data loaded successfully.")

            # Creating the dataset   
            logger.info("Creating the Dataset from loaded training data") 
            train_dataset = DrowsinessDataset(data.X_train, data.y_train)
            val_dataset = DrowsinessDataset(data.X_val, data.y_val)

            # Creating the dataloader
            logger.info("Creating the DataLoader for training data.")
            self.train_dataloader = DataLoader(dataset = train_dataset,
                                                batch_size = self.config.batch_size,
                                                shuffle = True)
            
            logger.info("Creating the DataLoader for validation data.")
            self.test_dataloader = DataLoader(dataset = val_dataset,
                                            batch_size = self.config.batch_size,
                                            shuffle = False)
            
            # Initializing the model
            logger.info("Initializing the DrowsinessDetectorModel.")
            model = DrowsinessDetectorModel(self.config)
            model.create_model()

            logger.info("Model initialized successfully.")
            # Move model to the selected device
            model.to(self.select_device())

            # Start model training
            logger.info("Starting the training loop")
            self.train_and_val_loop(model, data)

            # Save the trained model
            self.save_model(model)


        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise e


class DrowsinessDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
        return self.X[idx], self.y[idx]


class DrowsinessDetectorModel(nn.Module):
    def __init__(self, training_config: TrainingConfig):
        super(DrowsinessDetectorModel, self).__init__()
        self.config = training_config
        self.input_size = self.config.no_features
        self.hidden_size = self.config.no_lstm_units
        self.num_layers = self.config.no_lstm_layers
        self.dropout = self.config.dropout


    def create_model(self):
        """
        Creates the LSTM model for drowsiness detection.
        
        """
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.dropout)
        
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, X):
        """
        Forward pass of the model.
        
        """
        out, _ = self.lstm(X)
        out = out[:, -1, :] # Get the last time step output
        out = self.fc(out)
        return out
