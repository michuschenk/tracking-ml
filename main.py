# Example based on:
# https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from trackers import generate_tracking_data
import visualise as vis

# (0) CONFIG.
tracker = 'nonlinear'

generate_training_set = True
n_particles_train = 50000
# Misleading name: size of training dataset, only 1 turn used for training
n_turns_train = 10
visualise_training_data = True

# NN configuration
n_nodes_NN = 15
n_hidden_layers = 1   # >= 1
visualise_NN_training = True

generate_test_set_1turn = False
generate_test_set_multiturn = False
n_particles_test_1turn = 100
n_particles_test_multiturn = 1000
n_turns_test_multiturn = 2
visualise_test_results = True

# (1) GENERATE TRAINING DATA USING EITHER LIN. TRACKER OR PYSIXTRACKLIB
# Note that part of training set will be used as validation set during
# training.
# TODO: is Gaussian the best choice for training data?
#  (maybe for NN weights?)

# TODO: Linear tracking needs fixing ...
if tracker not in ['linear', 'nonlinear']:
    raise ValueError("Unknown input for 'tracking' -- must be either" +
                     "'linear' (smooth approx.) or 'nonlinear' " +
                     "(sixtracklib).")

filename = '{:s}_training_dataset.h5'.format(tracker)
if generate_training_set:
    # Generate new training data set and overwrite existing h5 file
    training_data = generate_tracking_data(
        tracker=tracker, n_particles=n_particles_train,
        n_turns=n_turns_train, filename=filename)
else:
    hdf_file = pd.HDFStore(filename, mode='r')
    training_data = hdf_file['data']

# (2) PREPARE DATA FOR TRAINING
# We train from first turn data only (i.e. train_X will be initial
# particle phase space coords. and train_y (targets) are phase space
# coords. after 1 turn.)
# TODO: Try to train from turn 1 to turn 2 (uses more 'matched' beam)
# (2a) Data frames, input and target
# From turn 0 -> 1, then from turn 1 -> 2, etc. ?
train_X = pd.DataFrame()
train_y = pd.DataFrame()

for i in range(n_turns_train):
    train_X_tmp = (training_data.loc[training_data['turn'] == i]
                   .sort_values('particle_id')
                   .drop(columns=['particle_id', 'turn']))
    train_y_tmp = (training_data.loc[training_data['turn'] == (i+1)]
                   .sort_values('particle_id')
                   .reset_index()
                   .drop(columns=['index', 'particle_id', 'turn']))

    # Select subsample of training data
    train_X_tmp = train_X_tmp[:n_particles_train]
    train_y_tmp = train_y_tmp[:n_particles_train]

    train_X = train_X.append(train_X_tmp)
    train_y = train_y.append(train_y_tmp)


if visualise_training_data:
    # Training data before scaling
    fig0 = plt.figure(0, figsize=(12, 6))
    plt.suptitle("Training data\n(before scaling)", fontsize=18)
    vis.input_vs_output_data(train_X, train_y, fig=fig0)

# (2b) Standardise input and target (important step). If e.g. output
# not standardised, NN does not train well.
# TODO: Input from Elena Fol: try to initialise weights in correct range
#  this should also solve the problem (instead of standardising output)
scaler_in = StandardScaler()
train_X = pd.DataFrame(
    data=scaler_in.fit_transform(train_X), columns=train_X.columns)

scaler_out = StandardScaler()
train_y = pd.DataFrame(
    data=scaler_out.fit_transform(train_y), columns=train_y.columns)

if visualise_training_data:
    # Training data after scaling
    fig1 = plt.figure(1, figsize=(12, 6))
    plt.suptitle("Training data\n(after scaling)", fontsize=18)
    vis.input_vs_output_data(
        train_X, train_y, fig=fig1, xlims=(-5, 5), ylims=(-5, 5),
        units=False)

# (3) BUILD MODEL: use Sequential network type
# (sequential is the simplest way to build model in Keras:
# we add layer by layer)
NN_tracker = Sequential()

# Input layer
# TODO: Try different activation function
# (e.g. relu, tanh (hyperbolic tangent activation)
# tanh -> does not work as well as relu)
n_input_nodes = train_X.shape[1]
NN_tracker.add(
    Dense(n_nodes_NN, activation='relu', input_shape=(n_input_nodes,),
          use_bias=True))

# Additional hidden layers
if n_hidden_layers < 1:
    raise ValueError("There must be at least 1 hidden layer.")

for l in range(n_hidden_layers-1):
    NN_tracker.add(Dense(n_nodes_NN, activation='relu'))

# Output layer
n_output_nodes = train_y.shape[1]
NN_tracker.add(Dense(n_output_nodes))  # activation='linear'

# (4) COMPILE MODEL
# Choose optimiser and loss function
# lr=1e-3 is default
# adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
#             amsgrad=False)
NN_tracker.compile(optimizer='adam', loss='mean_squared_error',
                   metrics=['acc'])

# (5) TRAIN MODEL
# Fitting of model in epochs: use EarlyStopping to cancel
# training in case model does not improve anymore before
# reaching end of max. number of epochs (patience=5 means:
# stop if model performance on validation 'does not change'
# for 5 epochs in a row)
early_stopping_monitor = EarlyStopping(patience=5)
training_history = NN_tracker.fit(
    train_X, train_y, validation_split=0.2, epochs=500,
    callbacks=[early_stopping_monitor])

# (5b) Visualise training evolution
if visualise_NN_training:
    training_history = pd.DataFrame(data=training_history.history)
    fig2 = plt.figure(2, figsize=(9, 6))
    plt.suptitle('NN training evolution', fontsize=18)
    vis.training_evolution_NN(training_history, fig=fig2)

# (6) TEST SET: MAKE PREDICTIONS WITH THE MODEL
# Create new data with *same* 'machine': try to track one turn with NN
# and compare to full tracking output
# TODO: Try also initial distributions that network was not trained for
# TODO: Might have to probe much larger phase space during training
#  otherwise we end up doing extrapolation, which is a problem

# (6a) Test for single-turn tracking
filename = '{:s}_test_dataset_1turn.h5'.format(tracker)
if generate_test_set_1turn:
    # Generate new training data set and overwrite existing h5 file
    test_data_1turn = generate_tracking_data(
        tracker=tracker, n_particles=n_particles_test_1turn, n_turns=2,
        filename=filename, xsize=1e-5, ysize=1e-5)
else:
    hdf_file = pd.HDFStore(filename, mode='r')
    test_data_1turn = hdf_file['data']


test_X = (test_data_1turn.loc[test_data_1turn['turn'] == 1]
          .sort_values('particle_id')
          .drop(columns=['particle_id', 'turn']))
test_y = (test_data_1turn.loc[test_data_1turn['turn'] == 2]
          .sort_values('particle_id')
          .reset_index()
          .drop(columns=['particle_id', 'turn', 'index']))

# Apply *exactly same* standardisation as done for training
# (the values used for rescaling / shift are stored in the 'scalers'
test_X = pd.DataFrame(
    data=scaler_in.transform(test_X), columns=test_X.columns)

# Predict with trained NN model, perform inverse scaling of output
# to have physical result directly comparable to tracking
prediction_NN = NN_tracker.predict(test_X)
prediction_NN = pd.DataFrame(
    data=scaler_out.inverse_transform(prediction_NN),
    columns=test_X.columns)

# # Transform input back ...
# test_X = pd.DataFrame(
#     data=scaler_in.inverse_transform(test_X), columns=test_X.columns)

if visualise_test_results:
    # Visualise results
    fig3 = plt.figure(3, figsize=(12, 6))
    plt.suptitle('Phase space after 1 turn\n(full tracking vs. NN)',
                 fontsize=18)
    vis.test_data_phase_space(prediction_NN, test_y, fig=fig3)

    # Particle-by-particle overlays
    fig4 = plt.figure(4, figsize=(15, 6))
    plt.suptitle('Particle-by-particle overlay after 1 turn\n' +
                 '(full tracking vs. NN)', fontsize=18)
    vis.test_data(prediction_NN, test_y, fig=fig4)

    # Compute particle-by-particle and coord.-by-coord. differences
    difference = prediction_NN - test_y
    fig5 = plt.figure(5, figsize=(9, 6))
    plt.suptitle('Particle-by-particle differences after 1 turn\n' +
                 '(full tracking vs. NN)', fontsize=18)
    vis.test_data_difference(difference, fig=fig5)


# (6b) Test for multi-turn tracking
filename = '{:s}_test_dataset_multiturn.h5'.format(tracker)
if generate_test_set_multiturn:
    # Generate new training data set and overwrite existing h5 file
    test_data_multiturn = generate_tracking_data(
        tracker=tracker, n_particles=n_particles_test_multiturn,
        n_turns=n_turns_test_multiturn, filename=filename, xsize=5e-5,
        ysize=5e-5)
else:
    hdf_file = pd.HDFStore(filename, mode='r')
    test_data_multiturn = hdf_file['data']


test_X = (test_data_multiturn.loc[test_data_multiturn['turn'] == 1]
          .sort_values('particle_id')
          .drop(columns=['particle_id', 'turn']))

test_y = (test_data_multiturn.loc[test_data_multiturn['turn'] == 3]
          .sort_values('particle_id')
          .reset_index()
          .drop(columns=['particle_id', 'turn', 'index']))

# Turn 1
# Apply *exactly same* standardisation as done for training
# (the values used for rescaling / shift are stored in the 'scalers'
test_X = pd.DataFrame(
    data=scaler_in.transform(test_X), columns=test_X.columns)

# Predict with trained NN model, perform inverse scaling of output
# to have physical result directly comparable to tracking
prediction_NN = NN_tracker.predict(test_X)
prediction_NN = pd.DataFrame(
    data=scaler_out.inverse_transform(prediction_NN),
    columns=test_X.columns)

# Turn 2
prediction_NN = pd.DataFrame(
    data=scaler_in.transform(prediction_NN), columns=test_X.columns)
prediction_NN = NN_tracker.predict(prediction_NN)
prediction_NN = pd.DataFrame(
    data=scaler_out.inverse_transform(prediction_NN),
    columns=test_X.columns)

# Visualise results
fig33 = plt.figure(33, figsize=(12, 6))
plt.suptitle('Phase space after 2 turns\n(full tracking vs. NN)',
             fontsize=18)
vis.test_data_phase_space(prediction_NN, test_y, fig=fig33)

"""
# From test_data_multi-turn extract turn-by-turn centroid. This will be
# the ground truth
test_y = (test_data_multiturn.groupby('turn')
          .mean()
          .drop(columns=['particle_id'])
          .sort_index())

# Predict with trained NN model
prediction_NN_centroid = np.zeros((n_turns_test_multiturn, 4))
prediction_NN = test_X.copy()
# Unfortunately need to do scaling of inputs and outputs every time...
for i in range(n_turns_test_multiturn):
    prediction_NN_centroid[i, :] = np.mean(prediction_NN, axis=0)
    prediction_NN = scaler_in.transform(prediction_NN)
    prediction_NN = NN_tracker.predict(prediction_NN)
    prediction_NN = scaler_out.inverse_transform(prediction_NN)
prediction_NN_centroid = pd.DataFrame(
    data={'x': prediction_NN_centroid[:, 0],
          'xp': prediction_NN_centroid[:, 1],
          'y': prediction_NN_centroid[:, 2],
          'yp': prediction_NN_centroid[:, 3]})

# Visualise turn-by-turn centroids and differences
if visualise_test_results:
    # Centroid overlays
    fig7 = plt.figure(7, figsize=(15, 6))
    plt.suptitle('Centroids for multi-turn tracking\n' +
                 '(full tracking vs. NN)', fontsize=18)
    vis.test_data(prediction_NN_centroid, test_y, fig=fig7,
                  xlabel='Turn')

    # Centroid differences
    difference = prediction_NN_centroid - test_y
    fig8 = plt.figure(8, figsize=(15, 6))
    plt.suptitle('Centroid differences for multi-turn tracking\n' +
                 '(full tracking vs. NN)', fontsize=18)
    vis.test_data_difference(difference, fig=fig8)
"""
