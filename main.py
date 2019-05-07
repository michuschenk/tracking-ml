# Example based on:
# https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from trackers import LinearTracker, PySixTrackLibTracker
import visualise as vis

# Set plot style
import seaborn as sns
sns.set(context='talk', font_scale=0.9)
sns.set_style('white')

# (0) CONFIG.
tracking = 'nonlinear'
n_particles = 1000
n_nodes_NN = 15
n_hidden_layers = 1   # >= 1
generate_training_set = True
generate_test_set = True
visualise_training_data = True

# (1) GENERATE TRAINING DATA USING EITHER LIN. TRACKER OR PYSIXTRACKLIB
# Note that part of training set will be used as validation set during
# training.
# TODO: is Gaussian the best choice for training data?
#  (maybe for NN weights?)

# TODO: Linear tracking needs fixing ...
if tracking not in ['linear', 'nonlinear']:
    raise ValueError("Unknown input for 'tracking' -- must be either" +
                     "'linear' (smooth approx.) or 'nonlinear' " +
                     "(sixtracklib).")

if generate_training_set:
    # Generate new training data set and overwrite existing h5 file
    if tracking == 'linear':
        lin_tracker = LinearTracker(beta_s0=25., beta_s1=25., Q=20.13)
        training_data = lin_tracker.create_dataset(
            n_particles=n_particles, distr='Gaussian', n_turns=10)
    elif tracking == 'nonlinear':
        nonlin_tracker = PySixTrackLibTracker()
        training_data = nonlin_tracker.create_dataset(
            n_particles=n_particles, n_turns=10)
    hdf_file = pd.HDFStore('{:s}_training_dataset.h5'.format(tracking),
                           mode='w')
    hdf_file['training_data'] = training_data
    hdf_file.close()
else:
    hdf_file = pd.HDFStore('{:s}_training_dataset.h5'.format(tracking),
                           mode='r')
    training_data = hdf_file['training_data']

# (2) PREPARE DATA FOR TRAINING
# We train from first turn data only (i.e. train_X will be initial
# particle phase space coords. and train_y (targets) are phase space
# coords. after 1 turn.)
# (2a) Data frames, input and target
train_X = (training_data.loc[training_data['turn'] == 0]
           .sort_values('particle_id')
           .drop(columns=['particle_id', 'turn']))
train_y = (training_data.loc[training_data['turn'] == 1]
           .sort_values('particle_id')
           .drop(columns=['particle_id', 'turn']))

if visualise_training_data:
    fig0 = plt.figure(0, figsize=(11, 6))
    plt.suptitle("1-turn training data\n(before scaling)", fontsize=18)
    vis.phase_space_data(train_X, train_y, fig=fig0)

# (2b) Standardise input and target (important step). If e.g. output
# not standardised, NN does not train well.
scaler_in = StandardScaler()
train_X = pd.DataFrame(
    data=scaler_in.fit_transform(train_X), columns=train_X.columns)

scaler_out = StandardScaler()
train_y = pd.DataFrame(
    data=scaler_out.fit_transform(train_y), columns=train_y.columns)

if visualise_training_data:
    fig1 = plt.figure(1, figsize=(11, 6))
    plt.suptitle("1-turn training data\n(after scaling)", fontsize=18)
    vis.phase_space_data(
        train_X, train_y, fig=fig1, xlims=(-5, 5), ylims=(-5, 5))

# (3) BUILD MODEL: use Sequential network type
# (sequential is the simplest way to build model in Keras:
# we add layer by layer)
NN_tracker = Sequential()

# Input layer
# TODO: Try different activation function
# (e.g. relu, tanh (hyperbolic tangent activation)
# tanh -> does not work as well as relu)
n_input_nodes = train_X.shape[1]   # 2 input nodes for 1D betatron, etc.
NN_tracker.add(
    Dense(n_nodes_NN, activation='relu', input_shape=(n_input_nodes,),
          use_bias=True, kernel_initializer='random_uniform'))

# Additional hidden layers
if n_hidden_layers < 1:
    raise ValueError("There must be at least 1 hidden layer.")

for l in range(n_hidden_layers-1):
    NN_tracker.add(Dense(n_nodes_NN, activation='relu'))

# Output layer
n_output_nodes = train_y.shape[1]  # 2 output nodes for 1D betatron, etc.
NN_tracker.add(Dense(n_output_nodes))  # activation='linear'

# (4) COMPILE MODEL
# Choose optimiser and loss function
NN_tracker.compile(optimizer='adam', loss='mean_squared_error')

# (5) TRAIN MODEL
# Fitting of model in epochs: use EarlyStopping to cancel
# training in case model does not improve anymore before
# reaching end of max. number of epochs (patience=5 means:
# stop if model does not change for 5 epochs in a row)
early_stopping_monitor = EarlyStopping(patience=10)
training_history = NN_tracker.fit(
    train_X, train_y, validation_split=0.2, epochs=500,
    callbacks=[early_stopping_monitor])

# (6) TEST SET: MAKE PREDICTIONS WITH THE MODEL
# Create new data with *same* 'machine': try to track one turn with NN
# and compare to full tracking output
# TODO: Try also initial distributions that network was not trained for
# TODO: Might have to probe much larger phase space during training
#  otherwise we end up doing extrapolation, which is a problem
if generate_test_set:
    # Generate new test data set and overwrite existing h5 file
    if tracking == 'linear':
        test_data_1turn = lin_tracker.create_dataset(
            n_particles=100, n_turns=1, distr='Gaussian')
    else:
        # Re-instantiate PySixTrackLibTracker, don't trust existing one.
        nonlin_tracker = PySixTrackLibTracker()
        test_data_1turn = nonlin_tracker.create_dataset(
            n_particles=100, n_turns=1)
    hdf_file = pd.HDFStore(
        '{:s}_test_dataset.h5'.format(tracking), mode='w')
    hdf_file['test_data_1turn'] = test_data_1turn
    hdf_file.close()
else:
    hdf_file = pd.HDFStore(
        '{:s}_test_dataset.h5'.format(tracking), mode='r')
    test_data_1turn = hdf_file['test_data_1turn']


test_X = (test_data_1turn.loc[training_data['turn'] == 0]
          .sort_values('particle_id')
          .drop(columns=['particle_id', 'turn']))
test_y = (test_data_1turn.loc[training_data['turn'] == 1]
          .sort_values('particle_id')
          .drop(columns=['particle_id', 'turn']))

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

# Transform back ...
test_X = pd.DataFrame(
    data=scaler_in.inverse_transform(test_X), columns=test_X.columns)

# Visualise results
# Phase space
fig2 = plt.figure(2, figsize=(11, 6))
plt.suptitle('1-turn tracking\n(full tracking vs. NN)', fontsize=18)
vis.phase_space_data(test_X, test_y, fig=fig2, reduced_plot=False)

"""
plt.plot(prediction_nn_df['x_NN'], prediction_nn_df['xp_NN'], c='r', ls='None',
         marker='x', ms=8, mew=2, label='NN')
plt.xlabel('x')
plt.ylabel('xp')
plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.legend()
plt.subplots_adjust(left=0.19, bottom=0.15, top=0.85, right=0.97)
plt.show()
"""

# plt.scatter(test_1Turn_df['x_out'], test_1Turn_df['xp_out'], c='r', s=60)
# plt.scatter(nn_pred_df['x_NN'], nn_pred_df['xp_NN'], c='b', s=40)
# # plt.ylim(-5e-7, 5e-7)
# # plt.xlim(-5e-5, 5e-5)
# plt.show()

# Compare to actual tracking output
# fig1 = plt.figure(1, figsize=(8, 8))
# ax1 = fig1.add_subplot(211)
# ax2 = fig1.add_subplot(212, sharex=ax1)
# nn_pred_df['x_NN'].plot(ax=ax1)
# nn_pred_df['xp_NN'].plot(ax=ax2)
# ax1.plot(test_y['x_out'], ls='--')
# ax2.plot(test_y['xp_out'], ls='--')
# # test_y['x_out'].plot(ax=ax1, ls='--')
# # test_y['xp_out'].plot(ax=ax2, ls='--')
# ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax1.set(ylabel='x')
# ax2.set(ylabel='xp', xlabel='Particle idx.')
# plt.tight_layout()
# plt.show()

"""
# Phase space
fig111 = plt.figure(111, figsize=(7, 7))
plt.suptitle('1-turn tracking, non-standardised', fontsize=18)
plt.plot(test_X[:, 0], test_X[:, 1], 'g.', label='Input')
plt.plot(test_y['x_out'], test_y['xp_out'], c='b', marker='o', ls='None',
         label='Tracker')
plt.plot(prediction_nn_df['x_NN'], prediction_nn_df['xp_NN'], c='r', ls='None',
         marker='x', ms=8, mew=2, label='NN')
plt.xlabel('x')
plt.ylabel('xp')
plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.legend()
plt.subplots_adjust(left=0.19, bottom=0.15, top=0.85, right=0.97)
plt.show()


# Plot differences
diff_df = pd.DataFrame()
diff_df['x_diff'] = (prediction_nn_df['x_NN'] - test_y['x_out'])
diff_df['xp_diff'] = (prediction_nn_df['xp_NN'] - test_y['xp_out'])

fig2 = plt.figure(2, figsize=(9, 6))
ax2 = fig2.add_subplot(111)
diff_df.plot(ax=ax2, title='Differences after 1 turn')
ax2.set(xlabel='Particle idx.', ylabel='Coord. diff. (NN - tracking)')
plt.tight_layout()
plt.show()
"""


"""
# TRACK 'SEVERAL' TURNS
n_turns = 10
if tracking == 'linear':
    test_manyTurns_df, centroid = lin_tracker.create_dataset(
        distr='Gaussian', n_particles=10000, n_turns=n_turns)
else:
    nonlin_tracker = PySixTrackLibTracker()
    test_manyTurns_df, centroid = nonlin_tracker.create_dataset(
        n_particles=100, n_turns=n_turns)


test_X = test_manyTurns_df.drop(columns=['x_out', 'xp_out'])
test_y = test_manyTurns_df.drop(columns=['x_in', 'xp_in'])

# Predict with trained NN model
centroid_pred = np.zeros((n_turns, 2))
nn_pred_y = test_X.copy()
# Unfortunately need to do scaling of inputs and outputs every time...
for i in range(n_turns):
    centroid_pred[i, :] = np.mean(nn_pred_y, axis=0)
    nn_pred_y = scaler_in.transform(nn_pred_y)
    nn_pred_y = NN_tracker_model.predict(nn_pred_y)
    nn_pred_y = scaler_out.inverse_transform(nn_pred_y)


# Compare turn-by-turn centroid
centroid_track_df = pd.DataFrame(
    data={'xbar_track': centroid[:, 0], 'xpbar_track': centroid[:, 1]})
centroid_NN_df = pd.DataFrame(
    data={'xbar_NN': centroid_pred[:, 0], 'xpbar_NN': centroid_pred[:, 1]})

fig3 = plt.figure(3, figsize=(8, 8))
plt.suptitle('Centroid evolution')
ax31 = fig3.add_subplot(211)
ax32 = fig3.add_subplot(212)
ax31.plot(centroid_track_df['xbar_track'], label='Tracking')
ax31.plot(centroid_NN_df['xbar_NN'], ls='--', label='NN')
ax32.plot(centroid_track_df['xbar_track'], label='Tracking')
ax32.plot(centroid_NN_df['xbar_NN'], ls='--', label='NN')
ax31.set(ylabel='x')
ax32.set(xlabel='Turn', ylabel='xp')
ax32.set_xlim(0, n_turns)
ax31.legend()
plt.tight_layout()
plt.show()

# Plot differences
fig31 = plt.figure(31, figsize=(8, 8))
plt.suptitle('Centroid differences vs. turns')
ax311 = fig31.add_subplot(211)
ax312 = fig31.add_subplot(212, sharex=ax311)
ax311.plot(centroid_NN_df['xbar_NN'] - centroid_track_df['xbar_track'])
ax312.plot(centroid_NN_df['xbar_NN'] - centroid_track_df['xbar_track'])
ax311.set(ylabel=r'$\Delta x$')
ax312.set(xlabel='Turn', ylabel=r"$\Delta x'$")
ax312.set_xlim(0, n_turns)
plt.tight_layout()
plt.show()
"""
