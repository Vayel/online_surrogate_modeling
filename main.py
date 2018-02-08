import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from skopt.learning import GaussianProcessRegressor

from functions import *
from samplers import jsonTransformOOS,jsonTransformSobol
from connector import evaluateModelOnInputs

import os
pwd = os.getcwd()


""" Default Algorithm Tuning Constants """
budget = 10#500

# Set out-of-sample test and montecarlo sizes
testSize = 2#100
monteCarlos = 2#100


# Get an on out-of-sample test set that does not have combinations from the
# batch or iterative experiments
finalTestSize = testSize * monteCarlos

# Set the ABM parameters and support

modelName = "Brock & Hommes -- ABM version"

parametersRange = {
  "beta": (0.0, 10.0),
  "demandIntercept": (0.0, 1.0),
  "demandSlope": (1.01, 1.1),
  "supplySlope": (1, 2.0),
  "w":(0.0, 1.0),
  "cost":(0.0, 5.0),
  "fitnessRationalAgents": (-2.0, 2.0),
  "fitnessNaiveAgents": (-2.0, 2.0)
}

outputName = "priceInTime"


# Loading the data if it exists, otherwise creating it
load_data = False

startMain = time.time()

if load_data:  # This is only for the budget = 500 setting
    XSobolJson = pd.read_csv('Data/BH/X.csv', index_col=0).values
    YSobolJson = pd.read_csv('Data/BH/y.csv', index_col=0).values
    XOSSJson = pd.read_csv('Data/BH/X_oos.csv', index_col=0).values
    YOSSJson = pd.read_csv('Data/BH/y_oos.csv', index_col=0).values
else:
    start = time.time()

    print("Generate Sobol samples for training set")
    XSobolJson = jsonTransformSobol(parametersRange, finalTestSize * budget)

    print("Evaluating Sobols example using Simudyne SDK")
    YSobolJson = evaluateModelOnInputs(modelName, 10, XSobolJson, outputName)

    print("Finished Evaluation on " + modelName + " for Sobol Examples")

    print("Saving in CSV files...")
    pd.DataFrame(XSobolJson).to_csv("Data/BH/X.csv")
    pd.DataFrame(YSobolJson).to_csv("Data/BH/y.csv")

    # At this point we have the Sobol sampled parameters and their ABM evaluations (i.e. GDP from the ABM).

    # Build Out-of-sample set
    print("Generate Out of Sample for training set")
    XOSSJson = jsonTransformOOS(parametersRange, finalTestSize * budget)

    # Removing duplicated examples
    XOSSJson = list(filter(lambda x: x not in XSobolJson, XOSSJson))

    print("Finished building OOS set")
    print("Running next step...")

    # Evaluate the test set for the ABM response
    print("Evaluating OOS example using Simudyne SDK")
    YOSSJson = evaluateModelOnInputs(modelName, 10, XOSSJson,outputName)

    # y_test = evaluate_islands_on_set(oos_set)

    print("Saving in CSV files...")
    pd.DataFrame(XOSSJson).to_csv("Data/BH/X_oos.csv")
    pd.DataFrame(YOSSJson).to_csv("Data/BH/y_oos.csv")

    end = time.time()
    print("Finished building test sets for ABM response in: ", end - start)

# Datasets conversions

def dictListToNpArray(dictList):
    return np.array([list(dict.values()) for dict in dictList])

XSobol = dictListToNpArray(XSobolJson)
YSobol = dictListToNpArray(YSobolJson)

XOSS = dictListToNpArray(XOSSJson)
YOSS = dictListToNpArray(YOSSJson)

# Reception and connection check
assert(XSobol.shape[0] == YSobol.shape[0])

# Compute the Kriging surrogate
print("Fitting the Kriging Model")
krigingModel = GaussianProcessRegressor(random_state=0)
krigingModel.fit(XSobol, YSobol)

# Compute the XGBoost surrogate
print("Fitting the XGBoost Model")
XGBoostModel = fitXGBoost(XSobol, YSobol)

# At this point, we have the XGBoost surrogate model.  What we need next is the bit which returns the parameterisations
# for positive calibrations.

nbModels = 2
predictions = [None] * nbModels

# This is not nice at all
print("Predicting on the two models")
predictions[0] = krigingModel.predict(XOSS).flatten()
predictions[1] = XGBoostModel.predict(XOSS).flatten()

print("Evaluating MSE Performance")
MSEperf = np.zeros((nbModels, monteCarlos))
for modelIndex in range(len(predictions)):
    for i in range(monteCarlos):
        MSEperf[modelIndex, i] = mean_squared_error(YOSS[i * testSize:(i + 1) * testSize],
                                                    predictions[int(modelIndex)][i * testSize:(i + 1) * testSize])


print("Plotting")

experiment_labels = ["Kriging", "XGBoost (Batch)"]

MSEperf = pd.DataFrame(MSEperf, index=experiment_labels)

krigingLabel = "Kriging: Mean " + '{:2.5f}'.format(MSEperf.iloc[0, :].mean()) + ", Variance " + '{:2.5f}'.format(
    MSEperf.iloc[0, :].var())
xgbLabel = "XGBoost: Mean " + '{:2.5f}'.format(MSEperf.iloc[1, :].mean()) + ", Variance " + '{:2.5f}'.format(
    MSEperf.iloc[1, :].var())

fig, ax = plt.subplots(figsize=(12, 5))
sns.distplot(MSEperf.iloc[0, :], label=krigingLabel, ax=ax)
sns.distplot(MSEperf.iloc[1, :], label=xgbLabel, ax=ax)

plt.title("Out-Of-Sample Prediction Performance for " + modelName)
plt.xlabel('Mean-Squared Error')
plt.yticks([])

plt.legend()

fileForPlots = "Plots/evaluation" + modelName + str(budget) + ".png"
print("Saving plots in file "+ fileForPlots)


fig.savefig(fileForPlots);

endMain = time.time()

print("Executed for in ", endMain - start, "s for budget ", budget)