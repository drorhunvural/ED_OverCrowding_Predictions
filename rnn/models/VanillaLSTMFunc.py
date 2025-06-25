from keras.layers import Input
from keras.models import Model
import pandas as pd
import numpy as np
import os
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model



class VanillaLSTMfunctional:
    def __init__(self):
        pass

    def build_model(self, input_shape, dropout_rate, lstm_units, optimizer, loss, learning_rate):
        """
        Build the Vanilla LSTM model.

        Parameters:
        - input_shape (tuple): The shape of the input data (time_steps, features).
        - dropout_rate (float): Dropout rate to use for regularization between LSTM layers.
        - lstm_units (int): Number of units in the LSTM layers.
        - optimizer (keras.optimizers.Optimizer): The optimizer to use during training.
        - loss (str or keras.losses.Loss): The loss function to use during training.
        - learning_rate (float): Learning rate to use during training.
        """
    

        inputs = Input(shape=input_shape)
        lstm1, state_h1, state_c1 = LSTM(units=lstm_units, return_sequences=True, return_state=True)(inputs)
        lstm1 = Dropout(dropout_rate)(lstm1)
        lstm2, state_h2, state_c2 = LSTM(units=lstm_units, return_sequences=True, return_state=True)(lstm1)
        lstm2 = Dropout(dropout_rate)(lstm2)
        lstm3, state_h3, state_c3 = LSTM(units=lstm_units, return_state=True)(lstm2)
        lstm3 = Dropout(dropout_rate)(lstm3)

        dense1 = Dense(units=64, activation='relu')(lstm3)
        dense2 = Dense(units=32, activation='relu')(dense1)
        dense3 = Dense(units=16, activation='relu')(dense2)

        output = Dense(units=1)(dense3)

        self.model = Model(inputs=inputs, outputs=output)

        # Compile the model with specified optimizer and loss function
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, patience, dropout_rate, lstm_units, optimizer, loss, learning_rate,weight_decay):
        """
        Train the Vanilla LSTM model.

        Parameters:
        - X_train (numpy.ndarray): Input data for training.
        - y_train (numpy.ndarray): Target data for training. X_train.shape[1:] => (80, 1)
        - X_val (numpy.ndarray): Input data for validation.
        - y_val (numpy.ndarray): Target data for validation.
        - epochs (int): Number of epochs to train the model.
        - batch_size (int): Batch size for training.
        - patience (int): Number of epochs with no improvement after which training will be stopped for early stopping.
        - dropout_rate (float): Dropout rate to use for regularization between LSTM layers.
        - lstm_units (int): Number of units in the LSTM layers.
        - optimizer (keras.optimizers.Optimizer): The optimizer to use during training.
        - loss (str or keras.losses.Loss): The loss function to use during training.
        - learning_rate (float): Learning rate to use during training.
        """
        
         # Create the optimizer based on the optimizer name
        if optimizer == 'Adam':
            optimizer = Adam(learning_rate=learning_rate, decay = weight_decay)
        elif optimizer == 'SGD':
            optimizer = SGD(learning_rate=learning_rate, decay = weight_decay)
        elif optimizer == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate, decay = weight_decay)
        else:
            raise ValueError("Unsupported optimizer: " + optimizer)

        self.build_model(X_train.shape[1:], dropout_rate, lstm_units, optimizer, loss, learning_rate)  # Build model with specified input shape and hyperparameters
        early_stopping = EarlyStopping(monitor='loss', patience=patience, verbose=1, restore_best_weights=True)
        self.history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])         



    def predict(self, X):
        """
        Make predictions using the Vanilla LSTM model.

        Parameters:
        - X (numpy.ndarray): Input data for prediction.

        Returns:
        - numpy.ndarray: Predicted values.
        """
        return self.model.predict(X)

    def save_model(self, model_dir=None, loss=None,dataset_name=None):
        """
        Save the trained model to a file with a dynamically generated name.

        Parameters:
        - model_dir (str): Directory to save the trained model. Default is current directory.
        - loss (float): Mean Absolute Error (loss). If provided, will be included in the filename.
        """
        if model_dir is None:
            model_dir = os.getcwd()  # Use the current working directory if model_dir is not provided

        # Construct filename based on loss
        if loss is not None:
            filename = f'reg_LSTMfunc_{dataset_name}_loss_{loss:.4f}.h5'
        else:
            filename = 'trained_model_LSTM_functional.h5'

        # Join directory and filename
        model_path = os.path.join(model_dir, filename)

        # Save the trained model
        self.model.save(model_path)
        print("Trained model saved to:", model_path)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model on test data and print evaluation metrics.

        Parameters:
        - X_test (numpy.ndarray): Input data for testing.
        - y_test (numpy.ndarray): Target data for testing.
        """
        # Predict on test data
        y_pred = self.predict(X_test)

        # Compute evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        # Print evaluation metrics
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("Mean Absolute Error (MAE):", mae)
