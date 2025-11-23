import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV, KFold

steel = "steel.csv"

independent_cols = ['normalising_temperature', 'tempering_temperature', 'percent_silicon', 'percent_chromium',
                    'percent_copper', 'percent_nickel', 'percent_sulphur', 'percent_carbon', 'percent_manganese']
dependent_col = 'tensile_strength'

df = pd.read_csv(steel)

X = df[independent_cols]
y = df[dependent_col]

neigh = KNeighborsRegressor()

cv_results = cross_validate(neigh, X, y, cv=10, scoring={"r2": "r2", "mse": "neg_mean_squared_error"},
                            return_train_score=True)

default_training_mse = -cv_results["train_mse"]
default_test_mse = -cv_results["test_mse"]
default_training_r2 = cv_results["train_r2"]
default_test_r2 = cv_results["test_r2"]

print("KNN REGRESSOR")
print("---------DEFAULT MODEL RESULTS---------")
print("Train MSE scores:", default_training_mse)
print("Mean Train MSE:", default_training_mse.mean())
print("Train R2 scores:", default_training_r2)
print("Mean Train R2:", default_training_r2.mean())
print()
print("Test MSE scores:", default_test_mse)
print("Mean Test MSE:", default_test_mse.mean())
print("Test R2 scores:", default_test_r2)
print("Mean Test R2:", default_test_r2.mean())

n_neighbour_values = [3, 6, 9, 12, 15]
weight_values=['uniform', 'distance']

training_best_R2_using_neighbours = 0
training_best_MSE_using_neighbours = 1000000
training_best_neighbours_R2 = 0
training_best_neighbours_MSE = 0

test_best_R2_using_neighbours = 0
test_best_MSE_using_neighbours= 1000000
test_best_neighbours_R2 = ""
test_best_neighbours_MSE = ""

print("\n ------------TESTING LEARNING_RATE------------- \n")
for n_neighbour in n_neighbour_values:
    neigh = KNeighborsRegressor(n_neighbors=n_neighbour)
    cv_results = cross_validate(neigh, X, y, cv=10, scoring={"r2": "r2", "mse": "neg_mean_squared_error"},
                                return_train_score=True)

    training_mse = -cv_results["train_mse"]
    test_mse = -cv_results["test_mse"]
    training_r2 = cv_results["train_r2"]
    test_r2 = cv_results["test_r2"]

    training_mean_mse = training_mse.mean()
    training_std_mse = training_mse.std(ddof=1)
    training_mean_r2 = training_r2.mean()
    training_std_r2 = training_r2.std(ddof=1)

    test_mean_mse = test_mse.mean()
    test_std_mse = test_mse.std(ddof=1)
    test_mean_r2 = test_r2.mean()
    test_std_r2 = test_r2.std(ddof=1)

    if training_mean_r2 > training_best_R2_using_neighbours:
        training_best_R2_using_neighbours = training_mean_r2
        training_best_neighbours_R2 = n_neighbour
    if training_mean_mse < training_best_MSE_using_neighbours:
        training_best_MSE_using_neighbours = training_mean_mse
        training_best_neighbours_MSE = n_neighbour

    if test_mean_r2 > test_best_R2_using_neighbours:
        test_best_R2_using_neighbours = test_mean_r2
        test_best_neighbours_R2 = n_neighbour
    if test_mean_mse < test_best_MSE_using_neighbours:
        test_best_MSE_using_neighbours = test_mean_mse
        test_best_neighbours_MSE = n_neighbour

    print('Learning Rate: ', n_neighbour)

    print('Mean MSE:', training_mean_mse)
    print('Std MSE:', training_std_mse)

    print('Mean R2:', training_mean_r2)
    print('Std R2:', training_std_r2)
    print()

    print('Mean MSE:', test_mean_mse)
    print('Std MSE:', test_std_mse)

    print('Mean R2:', test_mean_r2)
    print('Std R2:', test_std_r2)
    print()

print("N_Neighbours RESULTS")
print("Best Mean R2 on training:", training_best_R2_using_neighbours)
print("Best n_neighbors for R2 on training:", training_best_neighbours_R2)
print("Best Mean MSE on training:", training_best_MSE_using_neighbours)
print("Best n_neighbors for MSE on training:", training_best_neighbours_MSE)
print("Best Mean R2 on test:", test_best_R2_using_neighbours)
print("Best n_neighbors for R2 on test:", test_best_neighbours_R2)
print("Best Mean MSE on test:", test_best_MSE_using_neighbours)
print("Best n_neighbors for MSE on test:", test_best_neighbours_MSE)

# Testing MAX DEPTH
training_best_R2_using_weight = 0
training_best_MSE_using_weight = 1000000
training_best_weight_R2 = ""
training_best_weight_MSE = ""

test_best_R2_using_weight = 0
test_best_MSE_using_weight = 1000000
test_best_weight_R2 = ""
test_best_weight_MSE = ""

print("\n ------------TESTING WEIGHT------------- \n")
for weight in weight_values:
    neigh= KNeighborsRegressor(weights=weight)
    cv_results = cross_validate(neigh, X, y, cv=10, scoring={"r2": "r2", "mse": "neg_mean_squared_error"},
                                return_train_score=True)

    training_mse = -cv_results["train_mse"]
    test_mse = -cv_results["test_mse"]
    training_r2 = cv_results["train_r2"]
    test_r2 = cv_results["test_r2"]

    training_mean_mse = training_mse.mean()
    training_std_mse = training_mse.std(ddof=1)
    training_mean_r2 = training_r2.mean()
    training_std_r2 = training_r2.std(ddof=1)

    test_mean_mse = test_mse.mean()
    test_std_mse = test_mse.std(ddof=1)
    test_mean_r2 = test_r2.mean()
    test_std_r2 = test_r2.std(ddof=1)

    if training_mean_r2 > training_best_R2_using_weight:
        training_best_R2_using_weight = training_mean_r2
        training_best_weight_R2 = weight
    if training_mean_mse < training_best_MSE_using_weight:
        training_best_MSE_using_weight = training_mean_mse
        training_best_weight_MSE = weight

    if test_mean_r2 > test_best_R2_using_weight:
        test_best_R2_using_weight = test_mean_r2
        test_best_weight_R2 = weight
    if test_mean_mse < test_best_MSE_using_weight:
        test_best_MSE_using_weight = test_mean_mse
        test_best_weight_MSE = weight

    print('Max Depth: ', weight)

    print('Mean MSE:', training_mean_mse)
    print('Std MSE:', training_std_mse)

    print('Mean R2:', training_mean_r2)
    print('Std R2:', training_std_r2)
    print()

    # print('MSE Scores:', mse_scores)
    print('Mean MSE:', test_mean_mse)
    print('Std MSE:', test_std_mse)

    print('Mean R2:', test_mean_r2)
    print('Std R2:', test_std_r2)
    print()

print("WEIGHT RESULTS")
print("Best Mean R2 on training:", training_best_R2_using_weight)
print("Best Weight for R2 on training:", training_best_weight_R2)
print("Best Mean MSE on training:", training_best_MSE_using_weight)
print("Best Weight for MSE on training:", training_best_weight_MSE)
print("Best Mean R2 on test:", test_best_R2_using_weight)
print("Best Weight for R2 on test:", test_best_weight_R2)
print("Best Mean MSE on test:", test_best_MSE_using_weight)
print("Best Weight for MSE on test:", test_best_weight_MSE)

param_grid = {
    "n_neighbors": [3, 6, 9, 12, 15],
    "weights": ['uniform', 'distance']
}
cv = KFold(n_splits=10, shuffle=True, random_state=1)
neigh= KNeighborsRegressor()

grid = GridSearchCV(
    estimator=neigh,
    param_grid=param_grid,
    scoring={"r2": "r2", "mse": "neg_mean_squared_error"},
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
    refit=False
)

grid.fit(X, y)
res = grid.cv_results_

test_r2 = res["mean_test_r2"]
train_r2 = res["mean_train_r2"]
test_mse = -res["mean_test_mse"]
train_mse = -res["mean_train_mse"]

best_test_r2_idx = np.argmax(test_r2)
worst_test_r2_idx = np.argmin(test_r2)
best_test_mse_idx = np.argmin(test_mse)
worst_test_mse_idx = np.argmax(test_mse)

best_train_r2_idx = np.argmax(train_r2)
worst_train_r2_idx = np.argmin(train_r2)
best_train_mse_idx = np.argmin(train_mse)
worst_train_mse_idx = np.argmax(train_mse)

print("\n ------------TESTING BOTH N_NEIGHBOURS + WEIGHTS------------- \n")

print(" TRAINING RESULTS ")
print("Best Train R2:", train_r2[best_train_r2_idx], "| Params:", res["params"][best_train_r2_idx])
print("Worst Train R2:", train_r2[worst_train_r2_idx], "| Params:", res["params"][worst_train_r2_idx])
print("Best Train MSE (lowest):", train_mse[best_train_mse_idx], "| Params:", res["params"][best_train_mse_idx])
print("Worst Train MSE (highest):", train_mse[worst_train_mse_idx], "| Params:", res["params"][worst_train_mse_idx])
print(" TEST RESULTS ")
print("Best Test R2:", test_r2[best_test_r2_idx], "| Params:", res["params"][best_test_r2_idx])
print("Worst Test R2:", test_r2[worst_test_r2_idx], "| Params:", res["params"][worst_test_r2_idx])
print("Best Test MSE (lowest):", test_mse[best_test_mse_idx], "| Params:", res["params"][best_test_mse_idx])
print("Worst Test MSE (highest):", test_mse[worst_test_mse_idx], "| Params:", res["params"][worst_test_mse_idx])
print()

# params_df = pd.DataFrame(res["params"])
#
# # Attach metrics (convert neg MSE to positive)
# params_df["r2_train"] = res["mean_train_r2"]
# params_df["r2_test"] = res["mean_test_r2"]
# params_df["mse_train"] = -res["mean_train_mse"]
# params_df["mse_test"] = -res["mean_test_mse"]
#
# # Separate DataFrames for each plot
# df_train_r2 = params_df[["max_depth", "learning_rate", "r2_train"]].copy()
# df_test_r2 = params_df[["max_depth", "learning_rate", "r2_test"]].copy()
# df_train_mse = params_df[["max_depth", "learning_rate", "mse_train"]].copy()
# df_test_mse = params_df[["max_depth", "learning_rate", "mse_test"]].copy()
#
# # Pivot for heatmaps
# pivot_train_r2 = df_train_r2.pivot(index="max_depth", columns="learning_rate", values="r2_train")
# pivot_test_r2 = df_test_r2.pivot(index="max_depth", columns="learning_rate", values="r2_test")
# pivot_train_mse = df_train_mse.pivot(index="max_depth", columns="learning_rate", values="mse_train")
# pivot_test_mse = df_test_mse.pivot(index="max_depth", columns="learning_rate", values="mse_test")
#
# # Plot examples
# plt.figure(figsize=(8, 5))
# sns.heatmap(pivot_train_r2, annot=True, cmap="mako", fmt=".3f")
# plt.title("Gradient Boosting R2 (Train)")
# plt.ylabel("max_depth");
# plt.xlabel("learning_rate")
# plt.tight_layout();
# plt.show()
#
# plt.figure(figsize=(8, 5))
# sns.heatmap(pivot_test_r2, annot=True, cmap="mako", fmt=".3f")
# plt.title("Gradient Boosting R2 (Test)")
# plt.ylabel("max_depth");
# plt.xlabel("learning_rate")
# plt.tight_layout();
# plt.show()
#
# plt.figure(figsize=(8, 5))
# sns.heatmap(pivot_train_mse, annot=True, cmap="mako", fmt=".1f")
# plt.title("Gradient Boosting MSE (Train)")
# plt.ylabel("max_depth");
# plt.xlabel("learning_rate")
# plt.tight_layout();
# plt.show()
#
# plt.figure(figsize=(8, 5))
# sns.heatmap(pivot_test_mse, annot=True, cmap="mako", fmt=".1f")
# plt.title("Gradient Boosting MSE (Test)")
# plt.ylabel("max_depth");
# plt.xlabel("learning_rate")
# plt.tight_layout();
# plt.show()

#
#
#
#
#
#

