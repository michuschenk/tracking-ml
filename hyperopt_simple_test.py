import numpy as np
import pandas as pd
import hyperopt as hypo


# (1) Define objective function
def objective(x):
    """A function that returns scalar value, but can have as many input
    parameters as we like. It's the function we want to minimize using
    the Bayesian optimization technique."""

    # Create the polynomial object and evaluate at x
    f = np.poly1d([1, -2, -28, 28, 12, -26, 100])
    return f(x) * 0.05

# (2) Define domain space: The domain space is the input values over
#     which we want to search when optimizing.
domain_space = hypo.hp.uniform('x', -5, 6)


# (3) Choose optimization algorithm: We are using the Tree-structured
#     Parzen Estimator model, and we can have Hyperopt configure it for
#     us using the suggest method.
tpe_algo = hypo.tpe.suggest


# (4) Trials object: if we want to see evolution / progression of the
#     algorithm (not strictly necessary: hypyeropt does it internally
#     for us.
tpe_trials = hypo.Trials()


# (5) Run optimization (minimization)
# Run 2000 evals with the tpe algorithm
tpe_best = hypo.fmin(fn=objective, space=domain_space, algo=tpe_algo,
                     trials=tpe_trials, max_evals=2000)
print('Found optimum:', tpe_best)

# (6) Some further analysis of the evolution
tpe_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results],
                            'iteration': tpe_trials.idxs_vals[0]['x'],
                            'x': tpe_trials.idxs_vals[1]['x']})

tpe_results.head()
