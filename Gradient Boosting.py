import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV, KFold


steel="steel.csv"
independent_cols=['normalising_temperature', 'tempering_temperature', 'percent_silicon', 'percent_chromium', 'percent_copper', 'percent_nickel', 'percent_sulphur','percent_carbon','percent_manganese']
dependent_col='tensile_strength'

df = pd.read_csv(steel)
X = df[independent_cols]
y = df[dependent_col]
# random state set to 1 for reproducibility
cv = KFold(n_splits=10, shuffle=True, random_state=1)

gbr = GradientBoostingRegressor(random_state=1)
# I began using cross_val_scores but switched to cross_validate to capture the traininng scores as well
cv_results = cross_validate(gbr, X, y, cv=cv, scoring={"r2": "r2","mse": "neg_mean_squared_error"}, return_train_score=True)

default_training_mse = -cv_results["train_mse"]
default_test_mse = -cv_results["test_mse"]
default_training_r2 = cv_results["train_r2"]
default_test_r2 = cv_results["test_r2"]

print("Gradient Boosting Regressor ")
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


learning_rate_values = [0.01, 0.05, 0.1, 0.15, 0.2]
max_depth_values = [3, 4, 5, 6, 7]

# Testing LEARNING RATE
# Looking back all these for loops could have been done alot simpler using GridSearch
training_best_R2_using_LR=0
training_best_MSE_using_LR=1000000
training_best_LR_R2=""
training_best_LR_MSE=""

test_best_R2_using_LR=0
test_best_MSE_using_LR=1000000
test_best_LR_R2=""
test_best_LR_MSE=""

print("\n ------------TESTING LEARNING_RATE------------- \n")
for lr in learning_rate_values:
    gbr = GradientBoostingRegressor(learning_rate=lr, random_state=1)
    cv_results = cross_validate(gbr, X, y, cv=cv, scoring={"r2": "r2","mse": "neg_mean_squared_error"}, return_train_score=True)

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

    if training_mean_r2 > training_best_R2_using_LR:
        training_best_R2_using_LR=training_mean_r2
        training_best_LR_R2=lr
    if training_mean_mse < training_best_MSE_using_LR:
        training_best_MSE_using_LR=training_mean_mse
        training_best_LR_MSE=lr

    if test_mean_r2 > test_best_R2_using_LR:
        test_best_R2_using_LR=test_mean_r2
        test_best_LR_R2=lr
    if test_mean_mse < test_best_MSE_using_LR:
        test_best_MSE_using_LR=test_mean_mse
        test_best_LR_MSE=lr


    print('Learning Rate: ',lr )

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

print("LEARNING RATE RESULTS")
print("Best Mean R2 on training:", training_best_R2_using_LR)
print("Best Learning Rate for R2 on training:", training_best_LR_R2)
print("Best Mean MSE on training:", training_best_MSE_using_LR)
print("Best Learning Rate for MSE on training:", training_best_LR_MSE)
print("Best Mean R2 on test:", test_best_R2_using_LR)
print("Best Learning Rate for R2 on test:", test_best_LR_R2)
print("Best Mean MSE on test:", test_best_MSE_using_LR)
print("Best Learning Rate for MSE on test:", test_best_LR_MSE)



# Testing MAX DEPTH
training_best_R2_using_MD=0
training_best_MSE_using_MD=1000000
training_best_MD_R2=""
training_best_MD_MSE=""

test_best_R2_using_MD=0
test_best_MSE_using_MD=1000000
test_best_MD_R2=""
test_best_MD_MSE=""

print("\n ------------TESTING MAX_DEPTH------------- \n")
for md in max_depth_values:
    gbr = GradientBoostingRegressor(max_depth=md, random_state=1)
    cv_results = cross_validate(gbr, X, y, cv=cv, scoring={"r2": "r2","mse": "neg_mean_squared_error"}, return_train_score=True)

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

    if training_mean_r2 > training_best_R2_using_MD:
        training_best_R2_using_MD=training_mean_r2
        training_best_MD_R2=md
    if training_mean_mse < training_best_MSE_using_MD:
        training_best_MSE_using_MD=training_mean_mse
        training_best_MD_MSE=md

    if test_mean_r2 > test_best_R2_using_MD:
        test_best_R2_using_MD=test_mean_r2
        test_best_MD_R2=md
    if test_mean_mse < test_best_MSE_using_MD:
        test_best_MSE_using_MD=test_mean_mse
        test_best_MD_MSE=md


    print('Max Depth: ',md )

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

print("MAX Depth RESULTS")
print("Best Mean R2 on training:", training_best_R2_using_MD)
print("Best Max Depth for R2 on training:", training_best_MD_R2)
print("Best Mean MSE on training:", training_best_MSE_using_MD)
print("Best Max Depth for MSE on training:", training_best_MD_MSE)
print("Best Mean R2 on test:", test_best_R2_using_MD)
print("Best Max Depth for R2 on test:", test_best_MD_R2)
print("Best Mean MSE on test:", test_best_MSE_using_MD)
print("Best Max Depth for MSE on test:", test_best_MD_MSE)

# Testing BOTH LEARNING RATE + MAX DEPTH
# Switched to Gridsearch to use the paramgrid functionality
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    "max_depth": [3, 4, 5, 6, 7]
}
cv = KFold(n_splits=10, shuffle=True, random_state=1)
gbr = GradientBoostingRegressor(random_state=1)

grid = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    scoring={"r2": "r2", "mse": "neg_mean_squared_error"},
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
    refit=False
)

grid.fit(X, y)
res = grid.cv_results_

# didnt end up using stds
test_r2 = res["mean_test_r2"]
train_r2 = res["mean_train_r2"]
test_mse = -res["mean_test_mse"]
train_mse = -res["mean_train_mse"]

std_test_r2 = res["std_test_r2"]
std_train_r2 = res["std_train_r2"]
std_test_mse = res["std_test_mse"]
std_train_mse = res["std_train_mse"]

best_test_r2_idx = np.argmax(test_r2)
worst_test_r2_idx = np.argmin(test_r2)
best_test_mse_idx = np.argmin(test_mse)
worst_test_mse_idx = np.argmax(test_mse)

best_train_r2_idx = np.argmax(train_r2)
worst_train_r2_idx = np.argmin(train_r2)
best_train_mse_idx = np.argmin(train_mse)
worst_train_mse_idx = np.argmax(train_mse)

print("\n ------------TESTING BOTH LEARNING RATE + MAX_DEPTH------------- \n")

print(" TRAINING RESULTS ")
print("Best Train R2:", train_r2[best_train_r2_idx],  "| Params:", res["params"][best_train_r2_idx])
print("Worst Train R2:", train_r2[worst_train_r2_idx], "| Params:", res["params"][worst_train_r2_idx])
print("Best Train MSE (lowest):", train_mse[best_train_mse_idx],  "| Params:", res["params"][best_train_mse_idx])
print("Worst Train MSE (highest):", train_mse[worst_train_mse_idx], "| Params:", res["params"][worst_train_mse_idx])
print(" TEST RESULTS ")
print("Best Test R2:", test_r2[best_test_r2_idx],  "| Params:", res["params"][best_test_r2_idx])
print("Worst Test R2:", test_r2[worst_test_r2_idx],  "| Params:", res["params"][worst_test_r2_idx])
print("Best Test MSE (lowest):", test_mse[best_test_mse_idx], "| Params:", res["params"][best_test_mse_idx])
print("Worst Test MSE (highest):", test_mse[worst_test_mse_idx],  "| Params:", res["params"][worst_test_mse_idx])
print()

# plot a heatmap for both training and test R2 and MSE
params_df = pd.DataFrame(res["params"])

params_df["r2_train"] = res["mean_train_r2"]
params_df["r2_test"] = res["mean_test_r2"]
params_df["mse_train"] = -res["mean_train_mse"]
params_df["mse_test"] = -res["mean_test_mse"]

df_train_r2 = params_df[["max_depth", "learning_rate", "r2_train"]].copy()
df_test_r2 = params_df[["max_depth", "learning_rate", "r2_test"]].copy()
df_train_mse = params_df[["max_depth", "learning_rate", "mse_train"]].copy()
df_test_mse = params_df[["max_depth", "learning_rate", "mse_test"]].copy()

pivot_train_r2 = df_train_r2.pivot(index="max_depth", columns="learning_rate", values="r2_train")
pivot_test_r2 = df_test_r2.pivot(index="max_depth", columns="learning_rate", values="r2_test")
pivot_train_mse = df_train_mse.pivot(index="max_depth", columns="learning_rate", values="mse_train")
pivot_test_mse = df_test_mse.pivot(index="max_depth", columns="learning_rate", values="mse_test")

plt.figure(figsize=(8,5))
sns.heatmap(pivot_train_r2, annot=True, cmap="flare", fmt=".3f")
plt.title("Gradient Boosting R2 (Training)")
plt.ylabel("max_depth"); plt.xlabel("learning_rate")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(pivot_test_r2, annot=True, cmap="flare", fmt=".3f")
plt.title("Gradient Boosting R2 (Test)")
plt.ylabel("max_depth"); plt.xlabel("learning_rate")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(pivot_train_mse, annot=True, cmap="flare", fmt=".1f")
plt.title("Gradient Boosting MSE (Training)")
plt.ylabel("max_depth"); plt.xlabel("learning_rate")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(pivot_test_mse, annot=True, cmap="flare", fmt=".1f")
plt.title("Gradient Boosting MSE (Test)")
plt.ylabel("max_depth"); plt.xlabel("learning_rate")
plt.tight_layout(); plt.show()

