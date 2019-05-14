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
import visualise as vis


def tracker_model(train_X, train_y, val_X, val_y, scan_params):

    model = Sequential()

    leaky_relu = LeakyReLU(alpha=0.2)
    n_features = train_X.shape[1]
    model.add(
        Dense(scan_params['n_input_nodes'],
              activation=leaky_relu,
              input_shape=(n_features,),
              kernel_initializer=glorot_normal()))

    if scan_params['add_hidden_layer']:
        model.add(Dense(
            scan_params['n_hidden_nodes'],
            activation=leaky_relu,
            kernel_initializer=glorot_normal()))
    model.add(Dense(train_y.shape[1],
                    kernel_initializer=glorot_normal()))

    # GOOD ADAM PARAMETERS
    # adam_opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
    #                 decay=0.0001, amsgrad=False)
    adam_opt = Adam(lr=scan_params['adam_learning_rate'],
                    beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                    decay=scan_params['adam_decay'], amsgrad=False)

    model.compile(optimizer=adam_opt, loss='mean_squared_error',
                  metrics=['acc'])

    training_log = model.fit(
        train_X, train_y,
        epochs=1000,
        batch_size=scan_params['batch_size'],
        callbacks=[early_stopping_monitor])

    # (5b) Visualise training evolution
    if visualise_NN_training:
        training_log = pd.DataFrame(data=training_log.history)
        fig2 = plt.figure(2, figsize=(9, 6))
        plt.suptitle('NN training evolution', fontsize=18)
        vis.training_evolution_NN(training_log, fig=fig2)

# (0) CONFIG.
tracker = 'nonlinear'
fit_method = 'neural_network'

generate_training_set = False
n_particles_train = 200000
# Misleading name: size of training dataset, only 1 turn used for training
n_turns_train = 2
visualise_training_data = False
visualise_NN_training = True

generate_test_set_1turn = True
generate_test_set_multiturn = False
n_particles_test_1turn = 200
n_particles_test_multiturn = 100
n_turns_test_multiturn = 2
visualise_test_results = True

# (1) LOAD DATA
filename = 'nonlinear_training_dataset.h5'
hdf_file = pd.HDFStore(filename, mode='r')
training_data = hdf_file['data']

# (2) PREPARE FOR TRAINING
train_X = (training_data.loc[training_data['turn'] == 0]
               .sort_values('particle_id')
               .drop(columns=['particle_id', 'turn']))
train_y = (training_data.loc[training_data['turn'] == 1]
               .sort_values('particle_id')
               .reset_index()
               .drop(columns=['index', 'particle_id', 'turn']))

scaler_in = StandardScaler()
train_X = pd.DataFrame(
    data=scaler_in.fit_transform(train_X), columns=train_X.columns)

scaler_out = StandardScaler()
train_y = pd.DataFrame(
    data=scaler_out.fit_transform(train_y), columns=train_y.columns)
train_y = pd.DataFrame(
     data=train_y, columns=train_y.columns)




# Run prediction on training data to see if there is also prediction
# bias
train_y_pred = model.predict(train_X)
diff_train = train_y_pred - train_y
fig50 = plt.figure(50, figsize=(9, 6))
plt.suptitle('Particle-by-particle differences after 1 turn\n' +
             '(full tracking vs. NN), training set', fontsize=18)
vis.test_data_difference(diff_train, fig=fig50)
print('mean diff (training): ')
print("diff_train (x, x')", np.mean(diff_train['x']),
      np.mean(diff_train['xp']))
print("diff_train (y, y')", np.mean(diff_train['y']),
      np.mean(diff_train['yp']))


"""
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
        filename=filename, xsize=1e-7, ysize=1e-7)
else:
    hdf_file = pd.HDFStore(filename, mode='r')
    test_data_1turn = hdf_file['data']


test_X = (test_data_1turn.loc[test_data_1turn['turn'] == 0]
          .sort_values('particle_id')
          .drop(columns=['particle_id', 'turn']))
test_y = (test_data_1turn.loc[test_data_1turn['turn'] == 1]
          .sort_values('particle_id')
          .reset_index()
          .drop(columns=['particle_id', 'turn', 'index']))

# Apply *exactly same* standardisation as done for training
# (the values used for rescaling / shift are stored in the 'scalers'
test_X = pd.DataFrame(
    data=scaler_in.transform(test_X), columns=test_X.columns)

# Predict with trained NN model, perform inverse scaling of output
# to have physical result directly comparable to tracking
prediction = model.predict(test_X)
prediction = pd.DataFrame(
    data=scaler_out.inverse_transform(prediction),
    columns=test_X.columns)

# # Transform input back ...
# test_X = pd.DataFrame(
#     data=scaler_in.inverse_transform(test_X), columns=test_X.columns)

if visualise_test_results:
    # Visualise results
    fig3 = plt.figure(3, figsize=(12, 6))
    plt.suptitle('Phase space after 1 turn\n(full tracking vs. NN)',
                 fontsize=18)
    vis.test_data_phase_space(prediction, test_y, fig=fig3)

    # Particle-by-particle overlays
    fig4 = plt.figure(4, figsize=(15, 6))
    plt.suptitle('Particle-by-particle overlay after 1 turn\n' +
                 '(full tracking vs. NN)', fontsize=18)
    vis.test_data(prediction, test_y, fig=fig4)

    # Compute particle-by-particle and coord.-by-coord. differences
    difference = prediction - test_y
    fig5 = plt.figure(5, figsize=(9, 6))
    plt.suptitle('Particle-by-particle differences after 1 turn\n' +
                 '(full tracking vs. NN)', fontsize=18)
    vis.test_data_difference(difference, fig=fig5)
"""



"""
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
