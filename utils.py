""" Ignore Warnings """
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

""" Imports """
import numpy as np
import pandas as pd
from scipy.stats.distributions import entropy
import matplotlib.pylab as plt
import numba

# Gaussian Process Regression (Kriging)
# modified version of kriging to make a fair comparison with regard
# to the number of hyperparameter evaluations
from sklearn.gaussian_process import *# GaussianProcessRegressor


""" cross-validation
Cross validation is used in each of the rounds to approximate the selected
surrogate model over the data samples that are available.

The evaluated parameter combinations are randomly split into two sets. An
in-sample set and an out-of-sample set. The surrogate is trained and its
parameters are tuned to an in-sample set, while the out-of-sample performance
is measured (using a selected performance metric) on the out-of-sample set.
This out-of-sample performance is then used as a proxy for the performance
on the full space of unevaluated parameter combinations. In the case of the
proposed procedure, this full space is approximated by the randomly selected
pool.
"""
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
#from skopt import gp_minimize

""" performance metric """
# Mean Squared Error
from sklearn.metrics import mean_squared_error, f1_score

""" Defaults Algorithm Tuning Constants """
_N_EVALS = 10
_N_SPLITS = 5
_CALIBRATION_THRESHOLD = 1.00

""" Functions """
numba.jit()
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

numba.jit()
def evaluate_islands_on_set(parameter_combinations):
    y = np.zeros(parameter_combinations.shape[0])
    num_params = parameter_combinations.shape[1]

    if num_params == 1:
        for i, rho in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho,
                             _RNG_SEED=0)
            y[i] = calibration_measure(gdp)
    elif num_params == 2:
        for i, (rho, alpha) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha,
                             _RNG_SEED=0)
            y[i] = calibration_measure(gdp)

    elif num_params == 3:
        for i, (rho, alpha, phi) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha, phi=phi,
                             _RNG_SEED=0)
            y[i] = calibration_measure(gdp)

    elif num_params == 4:
        for i, (rho, alpha, phi, pi) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha, phi=phi,
                             pi=pi, _RNG_SEED=0)
            y[i] = calibration_measure(gdp)

    elif num_params == 5:
        for i, (rho, alpha, phi, pi, eps) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha, phi=phi,
                             pi=pi, eps=eps, _RNG_SEED=0)
            y[i] = calibration_measure(gdp)

    return y

numba.jit()
def island_abm(rho=0.01,
               alpha=1.5,
               phi=0.4,
               pi=0.4,
               eps=0.1,
               lambda_param=1,
               T=100,
               N=50,
               _RNG_SEED=0):
    """ Islands growth model

    Parameters
    ----------
    rho :
    alpha :
    phi : float, required
    eps :
    lambda_param: (Default = 1)
    T : int, required
    The number of periods for the simulation
    N : int, optional (Default = 50)
    Number of firms
    _RNG_SEED : int, optional (Default = 0)
    Random number seen
    Output
    ------
    GDP : array, length = [,T]
    Simulated GPD
    """
    # Set random number seed
    np.random.seed(_RNG_SEED)

    T_2 = int(T / 2)

    GDP = np.zeros((T, 1))

    # Distributions
    # Precompute random binomial draws
    xy = np.random.binomial(1, pi, (T, T))
    xy[T_2, T_2] = 1

    # Containers
    s = np.zeros((T, T))
    A = np.ones((N, 6))

    # Initializations
    A[:, 1] = T_2
    A[:, 2] = T_2
    m = np.zeros((T, T))
    m[T_2, T_2] = N
    dest = np.zeros((N, 2))

    """ Begin ABM Code """
    for t in range(T):
        w = np.zeros((N, N))
        signal = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i != j:
                    if A[j, 0] == 1:
                        w[i, j] = np.exp(-rho * (np.abs(A[j, 1] - A[i, 1]) + \
                                                 np.abs(A[j, 2] - A[i, 2])))

                        if np.random.rand() < w[i, j]:
                            signal[i, j] = s[int(A[j, 1]), int(A[j, 2])]

            if A[i, 0] == 1:
                A[i, 4] = s[int(A[i, 1]), int(A[i, 2])] * \
                          m[int(A[i, 1]), int(A[i, 2])] ** alpha
                A[i, 3] = s[int(A[i, 1]), int(A[i, 2])]

            if A[i, 0] == 3:
                A[i, 4] = 0
                rnd = np.random.rand()
                if rnd <= 0.25:
                    A[i, 1] += 1
                else:
                    if rnd <= 0.5:
                        A[i, 1] -= 1
                    else:
                        if rnd <= 0.75:
                            A[i, 2] += 1
                        else:
                            A[i, 2] -= 1

                if xy[int(A[i, 1]), int(A[i, 2])] == 1:
                    A[i, 0] = 1
                    m[int(A[i, 1]), int(A[i, 2])] += 1
                    if m[int(A[i, 1]), int(A[i, 2])] == 1:
                        s[int(A[i, 1]), int(A[i, 2])] = \
                            (1 + int(np.random.poisson(lambda_param))) * \
                            (A[i, 1] + A[i, 2]) + phi * A[i, 5] + np.random.randn()

            if (A[i, 0] == 1) and (np.random.rand() <= eps):
                A[i, 0] = 3
                A[i, 5] = A[i, 4]
                m[int(A[i, 1]), int(A[i, 2])] -= 1

            if t > T / 100:
                if A[i, 0] == 2:
                    A[i, 4] = 0
                    if dest[i, 0] != A[i, 1]:
                        if dest[i, 0] > A[i, 1]:
                            A[i, 1] += 1
                        else:
                            A[i, 1] -= 1
                    else:
                        if dest[i, 1] != A[i, 2]:
                            if dest[i, 1] > A[i, 2]:
                                A[i, 2] += 1
                            else:
                                A[i, 2] -= 1
                    if (dest[i, 0] == A[i, 1]) and (dest[i, 1] == A[i, 2]):
                        A[i, 0] = 1
                        m[int(dest[i, 0]), int(dest[i, 1])] += 1
                if A[i, 0] == 1:
                    best_sig = np.max(signal[i, :])
                    if best_sig > s[int(A[i, 1]), int(A[i, 2])]:
                        A[i, 0] = 2
                        A[i, 5] = A[i, 4]
                        m[int(A[i, 1]), int(A[i, 2])] -= 1
                        index = np.where(signal[i, :] == best_sig)[0]
                        if index.shape[0] > 1:
                            ind = int(index[int(np.random.uniform(0, len(index)))])
                        else:
                            ind = int(index)
                        dest[i, 0] = A[ind, 1]
                        dest[i, 1] = A[ind, 2]

        GDP[t, 0] = np.sum(A[:, 4])

    log_GDP = np.log(GDP)

    return log_GDP

numba.jit()
def calibration_measure(log_GDP):
    """ Calibration Measure

    Input
    -----

    GDP : array, required, length = [,T]

    Output
    ------

    agdp : float
    Average GDP growth rate

    """
    T = log_GDP.shape[0]
    log_GDP = log_GDP[~np.isinf(log_GDP)]
    log_GDP = log_GDP[~np.isnan(log_GDP)]
    if log_GDP.shape[0] > 0:
        GDP_growth_rate = (log_GDP[-1] - log_GDP[0]) / T
    else:
        GDP_growth_rate = 0

    return GDP_growth_rate

numba.jit()
def calibration_condition(average_GDP_growth_rate, threshold_condition):
    return average_GDP_growth_rate >= threshold_condition

numba.jit()
def custom_metric_regression(y_hat, y):
    return 'MSE', mean_squared_error(y.get_label(), y_hat)

numba.jit()
def custom_metric_binary(y_hat, y):
    return 'MSE', f1_score(y.get_label(), y_hat, average='weighted')

numba.jit()
def get_round_selections(evaluated_set_X, evaluated_set_y,
                         unevaluated_set_X,
                         predicted_positives, num_predicted_positives,
                         samples_to_select, calibration_threshold,
                         budget):
    samples_to_select = np.min([abs(budget - evaluated_set_y.shape[0]),
                                samples_to_select]).astype(int)

    if num_predicted_positives >= samples_to_select:
        round_selections = int(samples_to_select)
        selections = np.where(predicted_positives == True)[0]
        selections = np.random.permutation(selections)[:round_selections]

    elif num_predicted_positives <= samples_to_select:
        # select all predicted positives
        selections = np.where(predicted_positives == True)[0]

        # select remainder according to entropy weighting
        budget_shortfall = int(samples_to_select - num_predicted_positives)

        selections = np.append(selections,
                               get_new_labels_entropy(evaluated_set_X, evaluated_set_y,
                                                      unevaluated_set_X,
                                                      calibration_threshold,
                                                      budget_shortfall))

    else:  # if we don't have any predicted positive calibrations
        selections = get_new_labels_entropy(clf, unevaluated_set_X, samples_to_select)

    to_be_evaluated = unevaluated_set_X[selections]
    unevaluated_set_X = np.delete(unevaluated_set_X, selections, 0)
    evaluated_set_X = np.vstack([evaluated_set_X, to_be_evaluated])
    evaluated_set_y = np.append(evaluated_set_y, evaluate_islands_on_set(to_be_evaluated))

    return evaluated_set_X, evaluated_set_y, unevaluated_set_X

numba.jit()
def get_new_labels_entropy(evaluated_set_X, evaluated_set_y,
                           unevaluated_X, calibration_threshold,
                           number_of_new_labels):
    """ Get a set of parameter combinations according to their predicted label entropy
    """
    clf = fit_entropy_classifier(evaluated_set_X, evaluated_set_y, calibration_threshold)

    y_hat_probability = clf.predict_proba(unevaluated_X)
    y_hat_entropy = np.array(map(entropy, y_hat_probability))
    y_hat_entropy /= y_hat_entropy.sum()
    unevaluated_X_size = unevaluated_X.shape[0]

    selections = np.random.choice(a=unevaluated_X_size,
                                  size=number_of_new_labels,
                                  replace=False,
                                  p=y_hat_entropy)
    return selections

if __name__ == "__main__":
    print ("Utils Functions Imported successfully")
