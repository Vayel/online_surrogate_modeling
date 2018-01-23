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
import seaborn as sns
import numba

""" surrogate models """
# Xtreeme Gradient Boosted Decision Trees
from xgboost import *


numba.jit()
def set_surrogate_as_gbt():
    """ Set the surrogate model as Gradient Boosted Decision Trees
    Helper function to set the surrogate model and parameter space
    as Gradient Boosted Decision Trees.

    For detail, see:
    http://scikit-learn.org/stable/modules/generated/
    sklearn.ensemble.GradientBoostingRegressor.html
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


def fit_surrogate_model(X, y):
    """ Fit a surrogate model to the X,y parameter combinations
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
    """ Fit a surrogate model to the X,y parameter combinations
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
