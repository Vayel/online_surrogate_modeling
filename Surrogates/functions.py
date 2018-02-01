""" Ignore Warnings """
from Surrogates.islands import evaluate_islands_on_set, calibration_condition

import numpy as np
from scipy.stats.distributions import entropy
import numba
from xgboost import XGBRegressor, XGBClassifier

# Gaussian Process Regression (Kriging)
# modified version of kriging to make a fair comparison with regard
# to the number of hyperparameter evaluations


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
from skopt import gp_minimize

""" performance metric """
# Mean Squared Error
from sklearn.metrics import mean_squared_error, f1_score

""" Functions """
numba.jit()

clf = None


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


numba.jit()


def set_surrogate_as_gbt():
    """ Set the surrogate model as Gradient Boosted Decision Trees
    Helper function to set the surrogate model and parameter space
    as Gradient Boosted Decision Trees.
    For detail, see:
    http://scikit-learn.org/stable/modules/generated/
    sklearn.ensemble.GradientBoostingRegressor.html
    Parameters
    ----------
    None
    Returns
    -------
    surrogate_model :
    surrogate_parameter_space :
    """

    surrogate_model = XGBRegressor(seed=0)

    surrogate_parameter_space = [
        (100, 1000),  # n_estimators
        (0.01, 1),  # learning_rate
        (10, 1000),  # max_depth
        (0.0, 1),  # reg_alpha
        (0.0, 1),  # reg_lambda
        (0.25, 1.0)]  # subsample

    return surrogate_model, surrogate_parameter_space


numba.jit()


def custom_metric_regression(y_hat, y):
    return 'MSE', mean_squared_error(y.get_label(), y_hat)


numba.jit()


def custom_metric_binary(y_hat, y):
    return 'MSE', f1_score(y.get_label(), y_hat, average='weighted')


numba.jit()


def fit_surrogate_model(X, y):
    _N_EVALS = 10
    _N_SPLITS = 5

    """ Fit a surrogate model to the X,y parameter combinations
    Parameters
    ----------
    surrogate_model :
    X :
    y :
    Output
    ------
    surrogate_model_fitted : A surrogate model fitted
    """
    surrogate_model, surrogate_parameter_space = set_surrogate_as_gbt()

    def objective(params):
        n_estimators, learning_rate, max_depth, reg_alpha, \
        reg_lambda, subsample = params

        reg = XGBRegressor(n_estimators=n_estimators,
                           learning_rate=learning_rate,
                           max_depth=max_depth,
                           reg_alpha=reg_alpha,
                           reg_lambda=reg_lambda,
                           subsample=subsample,
                           seed=0)

        kf = KFold(n_splits=_N_SPLITS, random_state=0, shuffle=True)
        kf_cv = [(train, test) for train, test in kf.split(X, y)]

        return -np.mean(cross_val_score(reg,
                                        X, y,
                                        cv=kf_cv,
                                        n_jobs=1,
                                        fit_params={'eval_metric': custom_metric_regression},
                                        scoring="neg_mean_squared_error"))

    # use Gradient Boosted Regression to optimize the Hyper-Parameters.
    surrogate_model_tuned = gp_minimize(objective,
                                        surrogate_parameter_space,
                                        n_calls=_N_EVALS,
                                        acq_func='gp_hedge',
                                        n_jobs=-1,
                                        random_state=0, verbose=9)

    surrogate_model.set_params(n_estimators=surrogate_model_tuned.x[0],
                               learning_rate=surrogate_model_tuned.x[1],
                               max_depth=surrogate_model_tuned.x[2],
                               reg_alpha=surrogate_model_tuned.x[3],
                               reg_lambda=surrogate_model_tuned.x[4],
                               subsample=surrogate_model_tuned.x[5],
                               seed=0)

    surrogate_model.fit(X, y, eval_metric=custom_metric_regression)

    return surrogate_model


numba.jit()


def fit_entropy_classifier(X, y, calibration_threshold):
    _N_EVALS = 10
    _N_SPLITS = 5

    """ Fit a surrogate model to the X,y parameter combinations
    Parameters
    ----------
    surrogate_model :
    X :
    y :
    Output
    ------
    surrogate_model_fitted : A surrogate model fitted
    """
    y_binary = calibration_condition(y, calibration_threshold)
    _, surrogate_parameter_space = set_surrogate_as_gbt()

    def objective(params):
        n_estimators, learning_rate, max_depth, reg_alpha, \
        reg_lambda, subsample = params

        clf = XGBClassifier(n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            reg_alpha=reg_alpha,
                            reg_lambda=reg_lambda,
                            subsample=subsample,
                            seed=0,
                            objective="binary:logistic")

        skf = StratifiedKFold(n_splits=_N_SPLITS, random_state=0, shuffle=True)
        skf_cv = [(train, test) for train, test in skf.split(X, y_binary)]

        return -np.mean(cross_val_score(clf,
                                        X, y_binary,
                                        cv=skf_cv,
                                        n_jobs=1,
                                        fit_params={'eval_metric': custom_metric_binary},
                                        scoring="f1_weighted"))

    # use Gradient Boosted Regression to optimize the Hyper-Parameters.
    clf_tuned = gp_minimize(objective,
                            surrogate_parameter_space,
                            n_calls=_N_EVALS,
                            acq_func='gp_hedge',
                            n_jobs=-1,
                            random_state=0)

    clf = XGBClassifier(n_estimators=clf_tuned.x[0],
                        learning_rate=clf_tuned.x[1],
                        max_depth=clf_tuned.x[2],
                        reg_alpha=clf_tuned.x[3],
                        reg_lambda=clf_tuned.x[4],
                        subsample=clf_tuned.x[5],
                        seed=0)

    clf.fit(X, y_binary, eval_metric=custom_metric_binary)

    return clf


def get_round_selections(evaluated_set_X, evaluated_set_y,
                         unevaluated_set_X,
                         predicted_positives, num_predicted_positives,
                         samples_to_select, calibration_threshold,
                         budget, clf=None):
    """
    """
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
