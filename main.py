# Example based on:
# https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from helpers import LinearTracker, PySixTrackLibTracker

# Set plot style
import seaborn as sns
sns.set(context='talk', font_scale=1)
sns.set_style('white')

# (0) CONFIG.
tracking = 'linear'
n_particles = 100000
n_nodes_NN = 12
n_hidden_layers = 0
use_scaler = True
plot_training_data = False

# (1) GENERATE TRAINING DATA USING EITHER LIN. TRACKER OR PYSIXTRACKLIB
if tracking == 'linear':
    lin_tracker = LinearTracker(beta_s0=25., beta_s1=25., Q=20.13)
    train_df, _ = lin_tracker.create_dataset(
        n_particles=n_particles, distr='Gaussian', n_turns=1)
else:
    nonlin_tracker = PySixTrackLibTracker()
    train_df = nonlin_tracker.create_dataset(
        n_particles=n_particles, n_turns=1)


# (2) PREPARE DATA FOR TRAINING
# TODO: experiment with other normalisation/scaling techniques?
train_X = train_df.drop(columns=['x_out', 'xp_out'])
train_y = train_df.drop(columns=['x_in', 'xp_in'])

if plot_training_data:
    plt.figure(0)
    plt.suptitle('Training data, before scaling', fontsize=18)
    plt.plot(train_X['x_in'][::10], train_X['xp_in'][::10], 'or',
             label='input')
    plt.plot(train_y['x_out'][::10], train_y['xp_out'][::10], 'ob',
             label='output')
    plt.xlabel('x')
    plt.ylabel('xp')
    plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.legend()
    plt.subplots_adjust(left=0.2, bottom=0.15, top=0.85)
    plt.show()

if use_scaler:
    scaler_in = StandardScaler()
    scaler_out = StandardScaler()
    train_X = scaler_in.fit_transform(train_X)
    train_y = scaler_out.fit_transform(train_y)
    # plt.figure(10)
    # plt.suptitle('Training data, after scaling inputs')
    # plt.plot(train_X[:, 0], train_X[:, 1], '.r', label='input')
    # plt.plot(train_y['x_out'], train_y['xp_out'], '.b', label='output')
    # plt.legend()
    # plt.show()


# (3) BUILD MODEL: use Sequential model
# (sequential is the simplest way to build model in keras:
# we add layer by layer)
NN_tracker_model = Sequential()

# Input layer
# TODO: Try different activation function
# (e.g. relu, tanh (hyperbolic tangent activation)
# tanh -> does not work as well as relu)
n_input_nodes = train_X.shape[1]   # 2 input nodes for 1D betatron
NN_tracker_model.add(
    Dense(n_nodes_NN, activation='relu', input_shape=(n_input_nodes,),
          use_bias=True, kernel_initializer='random_uniform'))

# Middle layer
for l in range(n_hidden_layers):
    NN_tracker_model.add(Dense(n_nodes_NN, activation='relu'))

# Output layer
n_output_nodes = train_y.shape[1]  # 2 output nodes for 1D betatron
NN_tracker_model.add(Dense(n_output_nodes))  # activation='linear'

# (4) COMPILE MODEL
# Choose optimiser and loss function
NN_tracker_model.compile(optimizer='adam', loss='mean_squared_error')

# (5) TRAIN MODEL
# Fitting of model in epochs: use EarlyStopping to cancel
# training in case model does not improve anymore before
# reaching end of max. number of epochs (patience=5 means:
# stop if model does not change for 5 epochs in a row)
early_stopping_monitor = EarlyStopping(patience=10)
training_history = NN_tracker_model.fit(
    train_X, train_y, validation_split=0.2, epochs=500,
    callbacks=[early_stopping_monitor])

# (6) MAKE PREDICTIONS WITH THE MODEL
# Create new data with same 'machine': try to track with NN
# and compare to linear tracking matrix
if tracking == 'linear':
    test_1Turn_df, _ = lin_tracker.create_dataset(
        n_particles=100, n_turns=1, distr='Gaussian')
else:
    nonlin_tracker = PySixTrackLibTracker()
    test_1Turn_df = nonlin_tracker.create_dataset(
        n_particles=100, n_turns=1)

test_X = test_1Turn_df.drop(columns=['x_out', 'xp_out'])
test_y = test_1Turn_df.drop(columns=['x_in', 'xp_in'])

# Apply *exactly same* normalization as done for training!
# (the values used for rescaling / shift are stored in the 'scaler'
if use_scaler:
    test_X = scaler_in.transform(test_X)
    test_y = scaler_out.transform(test_y)

# Predict with trained NN model
nn_pred_y = NN_tracker_model.predict(test_X)
nn_pred_df = pd.DataFrame(
    data={'x_NN': nn_pred_y[:, 0], 'xp_NN': nn_pred_y[:, 1]})

# plt.scatter(test_1Turn_df['x_out'], test_1Turn_df['xp_out'], c='r', s=60)
# plt.scatter(nn_pred_df['x_NN'], nn_pred_df['xp_NN'], c='b', s=40)
# # plt.ylim(-5e-7, 5e-7)
# # plt.xlim(-5e-5, 5e-5)
# plt.show()

# Compare to actual tracking output
fig1 = plt.figure(1, figsize=(8, 8))
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212, sharex=ax1)
nn_pred_df['x_NN'].plot(ax=ax1)
nn_pred_df['xp_NN'].plot(ax=ax2)
ax1.plot(test_y[:, 0], ls='--')
ax2.plot(test_y[:, 1], ls='--')
# test_y['x_out'].plot(ax=ax1, ls='--')
# test_y['xp_out'].plot(ax=ax2, ls='--')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax1.set(ylabel='x')
ax2.set(ylabel='xp', xlabel='Particle idx.')
plt.tight_layout()
plt.show()

fig11 = plt.figure(11, figsize=(7, 7))
plt.suptitle('1-turn tracking, standardised data', fontsize=18)
plt.plot(test_X[:, 0], test_X[:, 1], 'g.', label='Input')
plt.plot(test_y[:, 0], test_y[:, 1], c='b', marker='o', ls='None',
         label='Tracker')
plt.plot(nn_pred_df['x_NN'], nn_pred_df['xp_NN'], c='r', ls='None',
         marker='x', ms=8, mew=2, label='NN')
plt.xlabel('x')
plt.ylabel('xp')
plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.legend()
plt.subplots_adjust(left=0.15, bottom=0.15, top=0.85)
plt.show()


# Inverse transform to get correct units back
test_X = scaler_in.inverse_transform(test_X)
test_y = scaler_out.inverse_transform(test_y)
nn_pred_df = scaler_out.inverse_transform(nn_pred_df)

fig111 = plt.figure(111, figsize=(7, 7))
plt.suptitle('1-turn tracking, non-standardised', fontsize=18)
plt.plot(test_X[:, 0], test_X[:, 1], 'g.', label='Input')
plt.plot(test_y[:, 0], test_y[:, 1], c='b', marker='o', ls='None',
         label='Tracker')
plt.plot(nn_pred_df[:, 0], nn_pred_df[:, 1], c='r', ls='None',
         marker='x', ms=8, mew=2, label='NN')
plt.xlabel('x')
plt.ylabel('xp')
plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.legend()
plt.subplots_adjust(left=0.15, bottom=0.15, top=0.85)
plt.show()

#
# # Plot differences
# diff_df = pd.DataFrame()
# diff_df['x_diff'] = (nn_pred_df['x_NN'] - test_y['x_out'])
# diff_df['xp_diff'] = (nn_pred_df['xp_NN'] - test_y['xp_out'])
#
# fig2 = plt.figure(2, figsize=(9, 6))
# ax2 = fig2.add_subplot(111)
# diff_df.plot(ax=ax2, title='Differences after 1 turn')
# ax2.set(xlabel='Particle idx.', ylabel='Coord. diff. (NN - tracking)')
# plt.tight_layout()
# plt.show()

"""
# TRACK 'SEVERAL' TURNS
n_turns = 200
if tracking == 'linear':
    test_manyTurns_df, centroid = lin_tracker.create_dataset(
        distr='Gaussian', n_particles=10000, n_turns=n_turns)
if tracking == 'nonlinear':



test_X = test_manyTurns_df.drop(columns=['x_out', 'xp_out'])
test_y = test_manyTurns_df.drop(columns=['x_in', 'xp_in'])

# Predict with trained NN model
centroid_pred = np.zeros((n_turns, 2))
nn_pred_y = test_X.copy()
for i in range(n_turns):
    centroid_pred[i, :] = np.mean(nn_pred_y, axis=0)
    # TODO: rescaling with same scaler needs to be done every time?
    nn_pred_y = scaler.transform(nn_pred_y)
    nn_pred_y = NN_tracker_model.predict(nn_pred_y)


# Compare turn-by-turn centroid
centroid_track_df = pd.DataFrame(
    data={'xbar_track': centroid[:, 0], 'xpbar_track': centroid[:, 1]})
centroid_NN_df = pd.DataFrame(
    data={'xbar_NN': centroid_pred[:, 0], 'xpbar_NN': centroid_pred[:, 1]})

fig3 = plt.figure(3, figsize=(9, 6))
ax3 = fig3.add_subplot(111)
centroid_track_df.plot(ax=ax3)
centroid_NN_df.plot(ax=ax3, ls='--')
ax3.set(xlabel='Turn', ylabel='Centroid (arb. units)')
plt.tight_layout()
plt.show()
"""
