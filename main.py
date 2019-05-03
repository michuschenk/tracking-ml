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
sns.set(context='talk', font_scale=1.2)
sns.set_style('white')

# (0) CONFIG.
tracking = 'nonlinear'
n_particles = 50000
n_nodes_NN = 12
n_middle_layers = 2

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
# TODO: experiment with other normalisation techniques?
train_X = train_df.drop(columns=['x_out', 'xp_out'])
train_y = train_df.drop(columns=['x_in', 'xp_in'])

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)

# (3) BUILD MODEL: use Sequential model
# (sequential is the simplest way to build model in keras:
# we add layer by layer)
NN_tracker_model = Sequential()

# Input layer
# TODO: Try more hidden layers
# TODO: Try different activation function
# (e.g. relu, tanh (hyperbolic tangent activation)
# tanh -> does not work as well as relu)
n_input_nodes = train_X.shape[1]   # 2 input nodes for 1D betatron
NN_tracker_model.add(
    Dense(n_nodes_NN, activation='relu', input_shape=(n_input_nodes,),
          use_bias=True, kernel_initializer='random_uniform'))

# Middle layer
for l in range(n_middle_layers):
    NN_tracker_model.add(Dense(n_nodes_NN, activation='relu'))

# Output layer
n_output_nodes = train_y.shape[1]  # 2 output nodes for 1D betatron
NN_tracker_model.add(Dense(n_output_nodes, activation='linear'))

# (4) COMPILE MODEL
# Choose optimiser and loss function
NN_tracker_model.compile(optimizer='adam', loss='mean_squared_error')

# (5) TRAIN MODEL
# Fitting of model in epochs: use EarlyStopping to cancel
# training in case model does not improve anymore before
# reaching end of max. number of epochs (patience=5 means:
# stop if model does not change for 5 epochs in a row)
early_stopping_monitor = EarlyStopping(patience=12)
training_history = NN_tracker_model.fit(
    train_X, train_y, validation_split=0.2, epochs=500,
    callbacks=[early_stopping_monitor])

# (6) MAKE PREDICTIONS WITH THE MODEL
# Create new data with same 'machine': try to track with NN
# and compare to linear tracking matrix
if tracking == 'linear':
    test_1Turn_df, _ = lin_tracker.create_dataset(
        n_particles=200, n_turns=1, distr='Gaussian')
else:
    nonlin_tracker = PySixTrackLibTracker()
    test_1Turn_df = nonlin_tracker.create_dataset(
        n_particles=200, n_turns=1)

test_X = test_1Turn_df.drop(columns=['x_out', 'xp_out'])
test_y = test_1Turn_df.drop(columns=['x_in', 'xp_in'])

# Apply *exactly same* normalization as done for training!
# (the values used for rescaling / shift are stored in the 'scaler'
test_X = scaler.transform(test_X)

# Predict with trained NN model
nn_pred_y = NN_tracker_model.predict(test_X)
nn_pred_df = pd.DataFrame(
    {'x_NN': nn_pred_y[:, 0], 'xp_NN': nn_pred_y[:, 1]})

# Compare to actual tracking output
fig1 = plt.figure(1, figsize=(9, 6))
ax1 = fig1.add_subplot(111)
nn_pred_df.plot(ax=ax1, title='NN prediction vs. tracker')
test_y.plot(ax=ax1, ls='--')
ax1.set(xlabel='Particle idx.', ylabel='Coord.')
plt.tight_layout()
plt.show()

# Plot differences
diff_df = pd.DataFrame()
diff_df['x_diff'] = (nn_pred_df['x_NN'] - test_y['x_out'])
diff_df['xp_diff'] = (nn_pred_df['xp_NN'] - test_y['xp_out'])

fig2 = plt.figure(2, figsize=(9, 6))
ax2 = fig2.add_subplot(111)
diff_df.plot(ax=ax2, title='Differences after 1 turn')
ax2.set(xlabel='Particle idx.', ylabel='Coord. diff. (NN - tracking)')
plt.tight_layout()
plt.show()

"""
# TRACK 'SEVERAL' TURNS
n_turns = 200
test_manyTurns_df, centroid = lin_tracker.create_dataset(
    distr='Gaussian', n_particles=10000, n_turns=n_turns)
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
