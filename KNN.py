import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


steel = "steel.csv"
independent_cols = ['normalising_temperature', 'tempering_temperature', 'percent_silicon', 'percent_chromium','percent_copper', 'percent_nickel', 'percent_sulphur', 'percent_carbon', 'percent_manganese']
dependent_col = 'tensile_strength'

df = pd.read_csv(steel)
X = df[independent_cols]
y = df[dependent_col]

neigh = KNeighborsRegressor()
# random state set to 1 for reproducibility
cv = KFold(n_splits=10, shuffle=True, random_state=1)
# I began using cross_val_scores but switched to cross_validate to capture the traininng scores as well
cv_results = cross_validate(neigh, X, y, cv=cv, scoring={"r2": "r2", "mse": "neg_mean_squared_error"},return_train_score=True)

default_training_mse = -cv_results["train_mse"]
default_test_mse = -cv_results["test_mse"]
default_training_r2 = cv_results["train_r2"]
default_test_r2 = cv_results["test_r2"]

print("KNN REGRESSOR")
print("---------DEFAULT MODEL RESULTS---------")
print("Mean Train R2:", default_training_r2.mean())
print("Train R2 Standard Deviation:", default_training_r2.std())
print("Mean Training MSE:", default_training_mse.mean())
print("Train MSE Standard Deviation:", default_training_mse.std())
print()
print("Mean Test R2:", default_test_r2.mean())
print("Test R2 Standard Deviation:", default_test_r2.std())
print("Mean Test MSE:", default_test_mse.mean())
print("Test MSE Standard Deviation:", default_test_mse.std())


n_neighbor_values = [3, 5, 7, 9, 11, 13, 15]
weight_values=['uniform', 'distance']

# Testing N_NEIGHBORS
# Looking back all these for loops could have been done alot simpler using GridSearch
training_best_R2_using_neighbours = 0
training_best_MSE_using_neighbours = 1000000
training_best_neighbours_R2 = 0
training_best_neighbours_MSE = 0

test_best_R2_using_neighbours = 0
test_best_MSE_using_neighbours= 1000000
test_best_neighbours_R2 = ""
test_best_neighbours_MSE = ""

print("\n ------------TESTING LEARNING_RATE------------- \n")
for n_neighbor in n_neighbor_values:
    neigh = KNeighborsRegressor(n_neighbors=n_neighbor)
    cv_results = cross_validate(neigh, X, y, cv=cv, scoring={"r2": "r2", "mse": "neg_mean_squared_error"},
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
        training_best_neighbours_R2 = n_neighbor
    if training_mean_mse < training_best_MSE_using_neighbours:
        training_best_MSE_using_neighbours = training_mean_mse
        training_best_neighbours_MSE = n_neighbor

    if test_mean_r2 > test_best_R2_using_neighbours:
        test_best_R2_using_neighbours = test_mean_r2
        test_best_neighbours_R2 = n_neighbor
    if test_mean_mse < test_best_MSE_using_neighbours:
        test_best_MSE_using_neighbours = test_mean_mse
        test_best_neighbours_MSE = n_neighbor

    print('Learning Rate: ', n_neighbor)

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

print("N_Neighbors RESULTS")
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
    cv_results = cross_validate(neigh, X, y, cv=cv, scoring={"r2": "r2", "mse": "neg_mean_squared_error"},
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

# Now testing both n_neighbors and weights together using GridSearchCV with preprocessing
# the knn__ is needed in front of the param names to indicate they are part of the KNN step in the pipeline
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    "knn__weights": ['uniform', 'distance']
}
cv = KFold(n_splits=10, shuffle=True, random_state=1)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor())
])

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring={"r2": "r2", "mse": "neg_mean_squared_error"},
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
    refit=False,
)

grid.fit(X, y)
res = grid.cv_results_

# didnt end up using stds
train_r2 = res["mean_train_r2"]
std_train_r2 = res["std_train_r2"]
test_r2 = res["mean_test_r2"]
std_test_r2 = res["std_test_r2"]

train_mse = -res["mean_train_mse"]
std_train_mse = -res["std_train_mse"]
test_mse = -res["mean_test_mse"]
std_test_mse = -res["std_test_mse"]

best_train_r2_idx = np.argmax(train_r2)
worst_train_r2_idx = np.argmin(train_r2)
best_train_mse_idx = np.argmin(train_mse)
worst_train_mse_idx = np.argmax(train_mse)

best_test_r2_idx = np.argmax(test_r2)
worst_test_r2_idx = np.argmin(test_r2)
best_test_mse_idx = np.argmin(test_mse)
worst_test_mse_idx = np.argmax(test_mse)

print("\n ------------TESTING BOTH N_NEIGHBORS + WEIGHTS (WITH PREPROCESSING)------------- \n")

print(" TRAINING RESULTS ")
print("Best Train R2:", train_r2[best_train_r2_idx], " | Params:", res["params"][best_train_r2_idx])
print("Worst Train R2:", train_r2[worst_train_r2_idx], " | Params:", res["params"][worst_train_r2_idx])
print("Best Train MSE (lowest):", train_mse[best_train_mse_idx], " | Params:", res["params"][best_train_mse_idx])
print("Worst Train MSE (highest):", train_mse[worst_train_mse_idx], " | Params:", res["params"][worst_train_mse_idx])

print(" TEST RESULTS ")
print("Best Test R2:", test_r2[best_test_r2_idx], " | Params:", res["params"][best_test_r2_idx])
print("Worst Test R2:", test_r2[worst_test_r2_idx], " | Params:", res["params"][worst_test_r2_idx])
print("Best Test MSE (lowest):", test_mse[best_test_mse_idx], " | Params:", res["params"][best_test_mse_idx])
print("Worst Test MSE (highest):", test_mse[worst_test_mse_idx], " | Params:", res["params"][worst_test_mse_idx])
print()

params_df = pd.DataFrame(res["params"])

# Rename knn__ params to simpler names for plotting
params_df.rename(columns={
    "knn__n_neighbors": "n_neighbors",
    "knn__weights": "weights"
}, inplace=True)

# plot a heatmap for both training and test R2 and MSE
params_df["r2_train"] = train_r2
params_df["r2_test"] = test_r2
params_df["mse_train"] = train_mse
params_df["mse_test"] = test_mse

df_train_r2 = params_df[["n_neighbors", "weights", "r2_train"]].copy()
df_test_r2 = params_df[["n_neighbors", "weights", "r2_test"]].copy()
df_train_mse = params_df[["n_neighbors", "weights", "mse_train"]].copy()
df_test_mse = params_df[["n_neighbors", "weights", "mse_test"]].copy()

pivot_train_r2 = df_train_r2.pivot(index="n_neighbors", columns="weights", values="r2_train")
pivot_test_r2 = df_test_r2.pivot(index="n_neighbors", columns="weights", values="r2_test")
pivot_train_mse = df_train_mse.pivot(index="n_neighbors", columns="weights", values="mse_train")
pivot_test_mse = df_test_mse.pivot(index="n_neighbors", columns="weights", values="mse_test")


plt.figure(figsize=(6, 4))
sns.heatmap(pivot_train_r2, annot=True, cmap="crest", fmt=".3f")
plt.title("KNN R2 (Training) - With Preprocessing")
plt.ylabel("n_neighbors")
plt.xlabel("weights")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(pivot_test_r2, annot=True, cmap="crest", fmt=".3f")
plt.title("KNN R2 (Test) - With Preprocessing")
plt.ylabel("n_neighbors")
plt.xlabel("weights")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(pivot_train_mse, annot=True, cmap="crest", fmt=".1f")
plt.title("KNN MSE (Training) - With Preprocessing")
plt.ylabel("n_neighbors")
plt.xlabel("weights")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(pivot_test_mse, annot=True, cmap="crest", fmt=".1f")
plt.title("KNN MSE (Test) - With Preprocessing")
plt.ylabel("n_neighbors")
plt.xlabel("weights")
plt.tight_layout()
plt.show()

