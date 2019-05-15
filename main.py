# Example based on:
# https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37

import pandas as pd
import talos as ta
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.initializers import glorot_normal
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import visualise as vis


def tracker_model(x_train, y_train, x_val, y_val, scan_params):

    model = Sequential()

    # leaky_relu = LeakyReLU(alpha=0.2)
    n_features = x_train.shape[1]
    model.add(
        Dense(scan_params['n_input_nodes'],
              activation='relu',
              input_shape=(n_features,),
              kernel_initializer=glorot_normal()))

    if scan_params['add_hidden_layer']:
        model.add(Dense(
            scan_params['n_hidden_nodes'],
            activation='relu',
            kernel_initializer=glorot_normal()))
    model.add(Dense(y_train.shape[1],
                    kernel_initializer=glorot_normal()))

    # GOOD ADAM PARAMETERS
    # adam_opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
    #                 decay=0.0001, amsgrad=False)
    adam_opt = Adam(lr=scan_params['adam_learning_rate'],
                    beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                    decay=scan_params['adam_decay'], amsgrad=False)

    model.compile(optimizer=adam_opt, loss='mean_squared_error',
                  metrics=['acc'])

    early_stopping_monitor = EarlyStopping(patience=20)
    training_log = model.fit(
        x_train, y_train,
        epochs=1000,
        batch_size=scan_params['batch_size'],
        callbacks=[early_stopping_monitor],
        validation_data=[x_val, y_val],
        verbose=0)

    return training_log, model


# (1) LOAD DATA AND PREPROCESS
n_samples = 100000
filename = 'nonlinear_training_dataset.h5'
hdf_file = pd.HDFStore(filename, mode='r')
training_data = hdf_file['data']

x_train = (training_data.loc[training_data['turn'] == 0]
           .sort_values('particle_id')
           .drop(columns=['particle_id', 'turn']))
y_train = (training_data.loc[training_data['turn'] == 1]
           .sort_values('particle_id')
           .reset_index()
           .drop(columns=['index', 'particle_id', 'turn']))

x_train = x_train[:n_samples]
y_train = y_train[:n_samples]

# Split training and validation sets
# train_X, val_X, train_y, val_y = train_test_split(
#     X, y, test_size=0.2, random_state=1)

# Standardise input and output
scaler_in = StandardScaler()
x_train = scaler_in.fit_transform(x_train)

scaler_out = StandardScaler()
y_train = scaler_out.fit_transform(y_train)


# (2) HYPERPARAMETER SCAN CONFIG.
scan_params = {
    'n_input_nodes': [10, 30, 100],
    'add_hidden_layer': [False, True],
    'n_hidden_nodes': [20],
    'adam_learning_rate': [5e-6, 1e-5, 5e-5, 1e-4, 5e-4],
    'adam_decay': [1e-6, 1e-5, 5e-5],
    'batch_size': [256, 512]
}

hyper_out = ta.Scan(x_train, y_train,
                    params=scan_params,
                    dataset_name='long_run',
                    model=tracker_model,
                    val_split=0.2)
