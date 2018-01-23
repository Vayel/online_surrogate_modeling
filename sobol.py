""" Ignore Warnings """
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

""" Imports """
import sobol_seq
import numba
import numpy as np
from utils import *


numba.jit()
def get_sobol_samples(n_dimensions, samples, parameter_support):
    # Get the range for the support
    support_range = parameter_support[:, 1] - parameter_support[:, 0]

    # Generate the Sobol samples
    random_samples = sobol_seq.i4_sobol_generate(n_dimensions, samples)

    # Compute the parameter mappings between the Sobol samples and supports
    sobol_samples = np.vstack([
        np.multiply(s, support_range) + parameter_support[:, 0]
        for s in random_samples])

    return sobol_samples

numba.jit()
def get_unirand_samples(n_dimensions, samples, parameter_support):
    # Get the range for the support
    support_range = parameter_support[:, 1] - parameter_support[:, 0]

    # Generate the Sobol samples
    random_samples = np.random.rand(n_dimensions,samples).T

    # Compute the parameter mappings between the Sobol samples and supports
    unirand_samples = np.vstack([
        np.multiply(s, support_range) + parameter_support[:, 0]
        for s in random_samples])

    return unirand_samples


print ("Sobol Functions Imported successfully")
if __name__ == "__main__":
    # Set the ABM Evaluation Budget
    budget = 500

    # Set out-of-sample test and montecarlo sizes
    test_size = 100
    montecarlos = 100

    # Get an on out-of-sample test set that does not have combinations from the
    # batch or iterative experiments
    final_test_size = (test_size * montecarlos)

    # Set the ABM parameters and support
    islands_exploration_range = np.array([
        (0.0, 10),  # rho
        (0.8, 2.0),  # alpha
        (0.0, 1.0),  # phi
        (0.0, 1.0),  # pi
        (0.0, 1.0)])  # eps

    param_dims = islands_exploration_range.shape[0]

    # Generate Sobol samples for training set
    n_dimensions = islands_exploration_range.shape[0]

    print("Build Sobol set: Generating sobol examples")
    evaluated_set_X_batch = get_sobol_samples(n_dimensions, budget, islands_exploration_range)

    print("Evaluate the Sobol set for the ABM response")
    evaluated_set_y_batch = evaluate_islands_on_set(evaluated_set_X_batch)

    print("Saving sobol results to CSV files")
    pd.DataFrame(evaluated_set_X_batch).to_csv("X.csv")
    pd.DataFrame(evaluated_set_y_batch).to_csv("y.csv")

    # Build Out-of-sample set
    print("Build Out-of-sample set: generation of uniform random examples")
    oos_set = get_unirand_samples(n_dimensions, final_test_size*budget, islands_exploration_range)
    selections = []
    for i, v in enumerate(oos_set):
        if (v not in evaluated_set_X_batch):
            selections.append(i)

    print("Removing duplicated examples")
    oos_set = unique_rows(oos_set[selections])[:final_test_size]

    # Evaluate the test set for the ABM response
    print("Evaluate the Out-of-sample set for the ABM response")
    y_test = evaluate_islands_on_set(oos_set)

    print("Saving Out-of-sample results to CSV files")
    pd.DataFrame(oos_set).to_csv("X_oos.csv")
    pd.DataFrame(y_test).to_csv("y_oos.csv")
