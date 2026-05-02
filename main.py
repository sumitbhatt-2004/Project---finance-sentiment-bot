import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def prepare_data(data, window_size=60):
    """
    Standardizing data for Deep Learning.
    window_size=60 means the model looks at the last 60 days to predict tomorrow.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    x_train, y_train = [], []

    for i in range(window_size, len(scaled_data)):
        x_train.append(scaled_data[i-window_size:i, 0])
        y_train.append(scaled_data[i, 0])
        
    return np.array(x_train), np.array(y_train), scaler

def build_lstm_model(input_shape):
    """
    Designing the Neural Network Architecture.
    """
    model = tf.keras.Sequential()

    # Layer 1: LSTM with Dropout to prevent overfitting
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.2))

    # Layer 2: Second LSTM layer
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))

    # Output Layer: Predicting a single price value
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
