import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score


steel="steel.csv"

independent_cols=['normalising_temperature', 'tempering_temperature', 'percent_silicon', 'percent_chromium', 'percent_copper', 'percent_nickel', 'percent_sulphur','percent_carbon','percent_manganese']
dependent_col='tensile_strength'

df = pd.read_csv(steel)

X = df[independent_cols]
y = df[dependent_col]



gbr = GradientBoostingRegressor(random_state=1)

mse_scores = cross_val_score(gbr, X, y, cv=10, scoring='neg_mean_squared_error')
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores, ddof=1)

r2_scores = cross_val_score(gbr, X, y, cv=10, scoring='r2')
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores, ddof=1)

print("---------DEFAULT MODEL RESULTS---------")
print('MSE Scores:', mse_scores)
print('Mean MSE:', mean_mse)
print('Std MSE:', std_mse)

print('\nR2 Score:', r2_scores)
print('Mean R2:', mean_r2)
print('Std R2:', std_r2)

learning_rate_values = [0.01, 0.05, 0.1, 0.15, 0.2]
max_depth_values = [3, 4, 5, 6, 7]

best_R2_using_LR=0
best_MSE_using_LR=0
best_LR_R2=""
best_LR_MSE=""

print("\n ------------TESTING LEARNING_RATE------------- \n")
for lr in learning_rate_values:
    gbr = GradientBoostingRegressor(learning_rate=lr, random_state=1)
    mse_scores = cross_val_score(gbr, X, y, cv=10, scoring='neg_mean_squared_error')
    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores, ddof=1)

    r2_scores = cross_val_score(gbr, X, y, cv=10, scoring='r2')
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores, ddof=1)

    if mean_r2 > best_R2_using_LR:
        best_R2_using_LR=mean_r2
        best_LR_R2=lr
    if mean_mse < best_MSE_using_LR:
        best_MSE_using_LR=mean_mse
        best_LR_MSE=lr


    print(f'Learning Rate: {lr}')
    # print('MSE Scores:', mse_scores)
    print('Mean MSE:', mean_mse)
    print('Std MSE:', std_mse)



    # print('R2 Score:', r2_scores)
    print('Mean R2:', mean_r2)
    print('Std R2:', std_r2)
    print()

print("Best Mean R2:", best_R2_using_LR)
print("Best Learning Rate for R2:", best_LR_R2)
print("Best Mean MSE:", best_MSE_using_LR)
print("Best Learning Rate for MSE:", best_LR_MSE)






best_R2_using_LR=0
best_MSE_using_LR=0
best_LR_R2=""
best_LR_MSE=""

