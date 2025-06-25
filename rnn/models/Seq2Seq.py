from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping
import numpy as np
import os

class Seq2SeqLSTM:
    def __init__(self):
        self.model = None

    def build_model(self, input_shape, lstm_units, dropout_rate, optimizer, loss):
        """
        Build the Encoder-Decoder LSTM model with given parameters.

        Parameters:
        - input_shape (tuple): Shape of the input data, i.e., (time_steps, features).
        - lstm_units (int): Number of LSTM units.
        - dropout_rate (float): Dropout rate for regularization.
        - optimizer (keras.optimizers): Optimizer to use.
        - loss (str): Loss function to use.
        """
        encoder_inputs = Input(shape=input_shape, name='encoder_inputs')
        encoder_lstm = LSTM(lstm_units, return_state=True, name='encoder_lstm')
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = RepeatVector(input_shape[0])(encoder_outputs)
        decoder_lstm = LSTM(lstm_units, return_sequences=True, name='decoder_lstm')
        decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dropout = Dropout(dropout_rate)(decoder_outputs)  # Adding dropout layer
        decoder_dense = TimeDistributed(Dense(input_shape[1], activation='linear'), name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_dropout)

        model = Model(inputs=encoder_inputs, outputs=decoder_outputs, name='encoder_decoder_lstm')
        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
        
        self.model = model

    # def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, patience, dropout_rate, lstm_units, optimizer, loss, learning_rate):
    #     """
    #     Configure and train the model with provided parameters.

    #     Parameters:
    #     - X_train, y_train, X_val, y_val: Training and validation data.
    #     - epochs, batch_size, patience: Training parameters.
    #     - dropout_rate, lstm_units: Model architecture parameters.
    #     - optimizer, loss, learning_rate: Compilation parameters.
    #     """
    #     # Build model with given parameters
    #     self.build_model((X_train.shape[1], X_train.shape[2]), lstm_units, dropout_rate, Adam(learning_rate=learning_rate), loss)

    #     # Early stopping monitor
    #     early_stopping = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True, verbose=1)

    #     # Train the model
    #     self.history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
   
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, patience, dropout_rate, lstm_units, optimizer_name, loss, learning_rate, weight_decay):
            """
            Configure and train the model with provided parameters.

            Parameters:
            - X_train, y_train, X_val, y_val: Training and validation data.
            - epochs, batch_size, patience: Training parameters.
            - dropout_rate, lstm_units: Model architecture parameters.
            - optimizer_name, loss, learning_rate: Compilation parameters.
            """
            # Choose the optimizer based on the name
            if optimizer_name == 'Adam':
                optimizer = Adam(learning_rate=learning_rate,decay=weight_decay)
            elif optimizer_name == 'SGD':
                optimizer = SGD(learning_rate=learning_rate,decay=weight_decay)
            elif optimizer_name == 'rmsprop':
                optimizer = RMSprop(learning_rate=learning_rate,decay=weight_decay)
            else:
                raise ValueError("Unsupported optimizer")

            # Build the model with the chosen parameters
            self.build_model((X_train.shape[1], X_train.shape[2]), lstm_units, dropout_rate, optimizer, loss)

            # Early stopping monitor
            early_stopping = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True, verbose=1) #Displays a message when early stopping is triggered (verbose=1).

            # Train the model
            self.history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

    def predict(self, X):
        """
        Predict using the trained model.
        """
        predictions = self.model.predict(X)
        return predictions

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
            filename = f'reg_Seq2Seq_{dataset_name}_trained_model_loss_{loss:.4f}.h5'
        else:
            filename = 'trained_model_lstm_seq2seq.h5'

        # Join directory and filename
        model_path = os.path.join(model_dir, filename)

        # Save the trained model
        self.model.save(model_path)
        print("Trained model saved to:", model_path)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test set.
        """
        results = self.model.evaluate(X_test, y_test, verbose=0)
        print('Test Loss:', results[0])
        print('Test MAE:', results[1])
        return {'loss': results[0], 'mae': results[1]}
