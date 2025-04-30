import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Input, BatchNormalization, Dropout, LayerNormalization, MultiHeadAttention, Flatten, Add
from keras.models import Sequential 
from keras.optimizers import Adam
from keras.models import Model


def train_model(model, X_train, Y_train, epochs=10, batch_size=32, \
                class_weights=None, X_val=None, Y_val=None):
    """
    Train a given Keras model using the provided training data. Use the 
    validation data to monitor the model during training.

    Parameters
    ----------
    model : keras.Sequential
        The Keras model to be trained.
    X_train : np.ndarray
        The input features for training the model.
    Y_train : np.ndarray
        The target values for training the model.
    epochs : int, optional
        The number of epochs to train the model for. Default is 10.
    batch_size : int, optional
        The batch size to use during training. Default is 32.
    class_weights : dict, optional
        A dictionary containing class weights to be applied to the loss function
        during training to balance the data. Default is None.
    X_val : np.ndarray, optional
        The input features for validating the model.
    y_val : np.ndarray, optional
        The target values for validating the model.

    Returns
    -------
    model : keras.Sequential
        The trained Keras model.
    history : keras.callbacks.History
        The training history of the model.
    """
    # Train the model
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, 
                        class_weight=class_weights, 
                        validation_data=(X_val, Y_val),
                        shuffle=True)
    return model, history


def build_baseline(input_shape):
    """
    Build a multi-class classification baseline using Dense layers.

    Parameters
    ----------
    input_shape : int
        The shape of the input data. This should be an integer
        specifying the number of features in the input data (e.g. 200).

    Returns
    -------
    baseline : keras.models.Sequential
        A Keras sequential model object representing the built model.
        The model architecture should consist of at least two Dense layers with
        appropriate sizes and activations, followed by an output layer with a 4 
        units and softmax. 
        The model should be compiled with the sparse categorical cross-entropy loss function, an adam optimizer, 
        and the accuracy metric.
    """
    # Build the model
    model = Sequential([
    # Added an Input layer to pass in the input shape https://keras.io/api/models/sequential/
        Input(shape=(input_shape, )),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(4, activation="softmax")
        ])
    
    # Compile the model with the necesarry parameters for a binary classification problem
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_deeper_mlp(input_shape):
  """
  Build a deeper classification model using Dense layers optimized for multi-class classification.

  Parameters
  ----------
  input_shape : int
      The number of features in the input data.

  Returns
  -------
  model : keras.models.Sequential
      A Keras sequential model object representing the built model.
  """
  
  # Build the model
  model = Sequential([
    Input(shape=(input_shape, )),
    Dense(4096, activation="relu"),
    BatchNormalization(),
    Dropout(0.8),
    Dense(2048, activation="relu"),
    BatchNormalization(),
    Dropout(0.6),
    Dense(1024, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(4, activation="softmax")
  ])

  # Compile the model)
  model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

  return model


def build_model(input_shape):
    

    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(4, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_resnet_mlp(input_shape):
    

    inputs = Input(shape=(input_shape,))
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    skip = x  # save for skip connection

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, skip])  # skip connection
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(4, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_transformer_encoder(input_shape):
    

    inputs = Input(shape=(input_shape,))
    x = Dense(128, activation='relu')(inputs)
    x = LayerNormalization()(x)
    
    x_attn = MultiHeadAttention(num_heads=4, key_dim=32)(x[:, None, :], x[:, None, :])  # self-attention
    x = Add()([x[:, None, :], x_attn])  
    x = LayerNormalization()(x)
    
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(4, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

