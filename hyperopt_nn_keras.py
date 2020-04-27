import hyperopt as hypo
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_normal
from keras import backend as K

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_data():
    """Provide training and independent test data."""

    # Load training data
    filename = 'nonlinear_training_dataset.h5'
    hdf_file = pd.HDFStore(filename, mode='r')
    training_data = hdf_file['data']

    i_turn_train = 0
    f_turn_train = 1
    n_particles = 5000

    x = pd.DataFrame()
    y = pd.DataFrame()
    for i in range(i_turn_train, f_turn_train):
        x_tmp = (training_data.loc[training_data['turn'] == i]
                 .sort_values('particle_id')
                 .drop(columns=['particle_id', 'turn']))
        y_tmp = (training_data.loc[training_data['turn'] == (i + 1)]
                 .sort_values('particle_id')
                 .reset_index()
                 .drop(columns=['index', 'particle_id', 'turn']))
        x = x.append(x_tmp[:n_particles])
        y = y.append(y_tmp[:n_particles])
    hdf_file.close()

    # Split into training and validation sets (note that here we don't
    # use the test set yet at all. It should always be used only at the
    # end when the model has been trained, validated, and finalised.
    x_train, x_test, y_train, y_test = (
        train_test_split(x, y, test_size=0.2))

    # Standardise input and output training data
    scaler_in = StandardScaler()
    x_train = pd.DataFrame(
        data=scaler_in.fit_transform(x_train), columns=x.columns)

    scaler_out = StandardScaler()
    y_train = pd.DataFrame(
        data=scaler_out.fit_transform(y_train), columns=y.columns)

    # Use scalars to also scale test data (no fitting here!)
    x_test = pd.DataFrame(
        data=scaler_in.transform(x_test), columns=x.columns)
    y_test = pd.DataFrame(
        data=scaler_out.transform(y_test), columns=y.columns)

    return x_train, y_train, x_test, y_test


def train_model(x_train, y_train, hypo_params):
    """A function that returns 1 value, but can have as many input
    parameters as we like. It's the function we want to minimize using
    the Bayesian optimization technique. Here, we want to run the NN
    training and return the loss on the validation set: this is our
    minimization problem."""

    model = Sequential()
    n_input_nodes = x_train.shape[1]
    leaky_relu = LeakyReLU(alpha=0.1)
    model.add(
        Dense(hypo_params['n_nodes'],
              activation=leaky_relu,
              input_shape=(n_input_nodes,),
              kernel_initializer=glorot_normal(seed=0)))
    n_output_nodes = y_train.shape[1]
    model.add(
        Dense(n_output_nodes,
              kernel_initializer=glorot_normal(seed=0)))

    adam_opt = Adam(lr=hypo_params['learning_rate'],
                    decay=hypo_params['decay'])
    model.compile(optimizer=adam_opt,
                  loss='mean_squared_error',
                  metrics=['acc'])

    early_stopping_monitor = EarlyStopping(patience=20)
    result = model.fit(x_train, y_train,
                       validation_split=0.2,
                       epochs=2000,
                       batch_size=hypo_params['batch_size'],
                       verbose=4,
                       callbacks=[early_stopping_monitor])
    return model


def test_model(model, x_test, y_test):
    test_accuracy = model.evaluate(x_test, y_test)
    return test_accuracy


def objective(hypo_params):
    x_train, y_train, x_test, y_test = get_data()
    model = train_model(x_train, y_train, hypo_params)
    test_loss = test_model(model, x_test, y_test)
    K.clear_session()

    return {'loss': test_loss[0], 'status': hypo.STATUS_OK, 'model': model}


# Main code
domain_space = {
  'n_nodes': hypo.hp.randint('n_nodes', low=5, high=100),
  'learning_rate': hypo.hp.loguniform('learning_rate', low=-12, high=0),
  'decay': hypo.hp.loguniform('decay', low=-13, high=-8),
  'batch_size': hypo.hp.randint('batch_size', low=100, high=1000)
}

tpe_trials = hypo.Trials()
tpe_best = hypo.fmin(fn=objective, space=domain_space,
                     algo=hypo.tpe.suggest, trials=tpe_trials,
                     max_evals=200)
print('Found optimum:', tpe_best)
