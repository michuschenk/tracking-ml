# Example based on:
# https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from helpers import create_fake_dataset

# (1) GENERATE OR IMPORT TRAINING DATA
inp, out, _ = create_fake_dataset(n_samples=50000, distr='Uniform')
# inp, out, _ = create_fake_dataset(n_samples=40000, distr='Gaussian')
data_dict = {'x_in': inp[0,:], 'xp_in': inp[1,:],
             'x_out': out[0,:], 'xp_out': out[1,:]}
df = pd.DataFrame(data_dict)

# (2) PREPARE DATA FOR TRAINING
# TODO: data normalization necessary?
train_X = df.drop(columns=['x_out', 'xp_out'])
train_y = df.drop(columns=['x_in', 'xp_in'])

# (3) BUILD MODEL: use Sequential model
# (sequential is the simplest way to build model in keras:
# we add layer by layer)
n_nodes_NN = 7
NN_tracker_model = Sequential()

# Input layer
n_input_nodes = train_X.shape[1] # 2 input nodes for 1D betatron
NN_tracker_model.add(
    Dense(n_nodes_NN, activation='relu', input_shape=(n_input_nodes,),
          use_bias=True, kernel_initializer='random_uniform'))

# Middle layer
# NN_tracker_model.add(Dense(n_nodes_NN, activation='relu'))

# Output layer
n_output_nodes = train_y.shape[1] # 2 output nodes for 1D betatron
NN_tracker_model.add(Dense(n_output_nodes))

# (4) COMPILE MODEL
# Choose optimiser and loss function
NN_tracker_model.compile(optimizer='adam', loss='mean_squared_error')

# (5) TRAIN MODEL
# Fitting of model in epochs: use EarlyStopping to cancel
# training in case model does not improve anymore before
# reaching end of max. number of epochs (patience=3 means:
# stop if model does not change for 3 epochs in a row)
early_stopping_monitor = EarlyStopping(patience=6)
training_history = NN_tracker_model.fit(
    train_X, train_y, validation_split=0.2, epochs=500,
    callbacks=[early_stopping_monitor])

# (6) MAKE PREDICTIONS WITH THE MODEL
# Create new data with same 'machine': try to track with NN
# and compare to linear tracking matrix
test_inp, test_out, _ = create_fake_dataset(distr='Gaussian', n_samples=200)
test_data_dict = {
    'x_in': test_inp[0,:], 'xp_in': test_inp[1,:],
    'x_track': test_out[0,:], 'xp_track': test_out[1,:]}
test_df = pd.DataFrame(test_data_dict)
test_X = test_df.drop(columns=['x_track', 'xp_track'])
test_y = test_df.drop(columns=['x_in', 'xp_in'])

# Predict with trained NN model
nn_pred_y = NN_tracker_model.predict(test_X)
nn_pred_df = pd.DataFrame(
    {'x_NN': nn_pred_y[:,0], 'xp_NN': nn_pred_y[:,1]})

# Compare to actual tracking output
ax = nn_pred_df.plot(title='NN prediction vs. tracker')
test_y.plot(ax=ax, ls='--')
ax.set(xlabel='Particle idx.', ylabel='Coord. x')
plt.tight_layout()
plt.show()

# Plot differences
diff_df = pd.DataFrame()
diff_df['x_diff'] = nn_pred_df['x_NN'] - test_y['x_track']
diff_df['xp_diff'] = nn_pred_df['xp_NN'] - test_y['xp_track']
ax2 = diff_df.plot(title='Differences after 1 turn')
ax2.set(xlabel='Particle idx.', ylabel='Coord. diff. (NN - tracking)')
plt.tight_layout()
plt.show()


# TRACK 'SEVERAL' TURNS
n_turns = 200
test_inp, test_out, centroid = create_fake_dataset(
    distr='Gaussian', n_samples=4000, n_turns=n_turns)
test_data_dict = {
    'x_in': test_inp[0,:], 'xp_in': test_inp[1,:],
    'x_track': test_out[0,:], 'xp_track': test_out[1,:]}
test_df = pd.DataFrame(test_data_dict)
test_X = test_df.drop(columns=['x_track', 'xp_track'])
test_y = test_df.drop(columns=['x_in', 'xp_in'])

# Predict with trained NN model
centroid_pred = np.zeros((n_turns, 2))
nn_pred_y = test_X.copy()
for i in range(n_turns):
    centroid_pred[i,:] = np.mean(nn_pred_y, axis=0)
    nn_pred_y = NN_tracker_model.predict(nn_pred_y)

# Compare turn-by-turn centroid
centroid_track_df = pd.DataFrame(
    data={'xbar_track': centroid[:,0], 'xpbar_track': centroid[:,1]})
centroid_NN_df = pd.DataFrame(
    data={'xbar_NN': centroid_pred[:,0], 'xpbar_NN': centroid_pred[:,1]})

ax3 = centroid_track_df.plot()
centroid_NN_df.plot(ax=ax3, ls='--')
ax3.set(xlabel='Turn', ylabel='Centroid (arb. units)')
plt.tight_layout()
plt.show()