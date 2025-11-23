import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate


steel="steel.csv"

independent_cols=['normalising_temperature', 'tempering_temperature', 'percent_silicon', 'percent_chromium', 'percent_copper', 'percent_nickel', 'percent_sulphur','percent_carbon','percent_manganese']
dependent_col='tensile_strength'

df = pd.read_csv(steel)

X = df[independent_cols]
y = df[dependent_col]



gbr = GradientBoostingRegressor(random_state=1)

cv_results = cross_validate(gbr, X, y, cv=10, scoring={"r2": "r2","mse": "neg_mean_squared_error"}, return_train_score=True)

default_training_mse = -cv_results["train_mse"]
default_test_mse = -cv_results["test_mse"]
default_training_r2 = cv_results["train_r2"]
default_test_r2 = cv_results["test_r2"]

print("Train MSE scores:", default_training_mse)
print("Mean Train MSE:", default_training_mse.mean())
print("Train R2 scores:", default_training_r2)
print("Mean Train R2:", default_training_r2.mean())
print()
print("Test MSE scores:", default_test_mse)
print("Mean Test MSE:", default_test_mse.mean())
print("Test R2 scores:", default_test_r2)
print("Mean Test R2:", default_test_r2.mean())



learning_rate_values = [0.01, 0.05, 0.1, 0.15, 0.2]
max_depth_values = [3, 4, 5, 6, 7]

# Testing Learning Rate
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
    cv_results = cross_validate(gbr, X, y, cv=10, scoring={"r2": "r2","mse": "neg_mean_squared_error"}, return_train_score=True)

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
    # print('MSE Scores:', mse_scores)
    print('Mean MSE:', training_mean_mse)
    print('Std MSE:', training_std_mse)



    # print('R2 Score:', r2_scores)
    print('Mean R2:', training_mean_r2)
    print('Std R2:', training_std_r2)
    print()

    # print('MSE Scores:', mse_scores)
    print('Mean MSE:', test_mean_mse)
    print('Std MSE:', test_std_mse)

    # print('R2 Score:', r2_scores)
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
    cv_results = cross_validate(gbr, X, y, cv=10, scoring={"r2": "r2","mse": "neg_mean_squared_error"}, return_train_score=True)

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
    # print('MSE Scores:', mse_scores)
    print('Mean MSE:', training_mean_mse)
    print('Std MSE:', training_std_mse)



    # print('R2 Score:', r2_scores)
    print('Mean R2:', training_mean_r2)
    print('Std R2:', training_std_r2)
    print()

    # print('MSE Scores:', mse_scores)
    print('Mean MSE:', test_mean_mse)
    print('Std MSE:', test_std_mse)

    # print('R2 Score:', r2_scores)
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



# Testing Both LEARNING RATE AND MAX DEPTH
training_best_R2_using_BOTH=0
training_best_MSE_using_BOTH=1000000
training_best_LR_R2=""
training_best_LR_MSE=""
training_best_MD_R2=""
training_best_MD_MSE=""
training_worst_R2_using_BOTH=1
training_worst_MSE_using_BOTH=0
training_worst_LR_R2=""
training_worst_LR_MSE=""
training_worst_MD_R2=""
training_worst_MD_MSE=""




test_best_R2_using_BOTH=0
test_best_MSE_using_BOTH=1000000
test_best_LR_R2=""
test_best_LR_MSE=""
test_best_MD_R2=""
test_best_MD_MSE=""
test_worst_R2_using_BOTH=1
test_worst_MSE_using_BOTH=0
test_worst_LR_R2=""
test_worst_LR_MSE=""
test_worst_MD_R2=""
test_worst_MD_MSE=""

print("\n ------------TESTING BOTH LEARNING RATE + MAX_DEPTH------------- \n")
for lr in learning_rate_values:
    for md in max_depth_values:
        gbr = GradientBoostingRegressor(max_depth=md, learning_rate=lr, random_state=1)
        cv_results = cross_validate(gbr, X, y, cv=10, scoring={"r2": "r2","mse": "neg_mean_squared_error"}, return_train_score=True)

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

        if training_mean_r2 > training_best_R2_using_BOTH:
            training_best_R2_using_BOTH=training_mean_r2
            training_best_MD_R2=md
            training_best_LR_R2=lr
        if training_mean_mse < training_best_MSE_using_BOTH:
            training_best_MSE_using_BOTH=training_mean_mse
            training_best_MD_MSE=md
            training_best_LR_MSE=lr
        if test_mean_r2 > test_best_R2_using_BOTH:
            test_best_R2_using_BOTH=test_mean_r2
            test_best_MD_R2=md
            test_best_LR_R2=lr
        if test_mean_mse < test_best_MSE_using_BOTH:
            test_best_MSE_using_BOTH=test_mean_mse
            test_best_MD_MSE=md
            test_best_LR_MSE=lr

        if training_mean_r2 < training_worst_R2_using_BOTH:
            training_worst_R2_using_BOTH=training_mean_r2
            training_worst_MD_R2=md
            training_worst_LR_R2=lr
        if training_mean_mse > training_worst_MSE_using_BOTH:
            training_worst_MSE_using_BOTH=training_mean_mse
            training_worst_MD_MSE=md
            training_worst_LR_MSE=lr
        if test_mean_r2 < test_worst_R2_using_BOTH:
            test_worst_R2_using_BOTH=test_mean_r2
            test_worst_MD_R2=md
            test_worst_LR_R2=lr
        if test_mean_mse > test_worst_MSE_using_BOTH:
            test_worst_MSE_using_BOTH=test_mean_mse
            test_worst_MD_MSE=md
            test_worst_LR_MSE=lr

        print('Max Depth: ',md, ' Learning Rate: ',lr )
        print('Mean MSE:', training_mean_mse)
        print('Std MSE:', training_std_mse)

        print('Mean R2:', training_mean_r2)
        print('Std R2:', training_std_r2)
        print()

        # print('MSE Scores:', mse_scores)
        print('Mean MSE:', test_mean_mse)
        print('Std MSE:', test_std_mse)

        # print('R2 Score:', r2_scores)
        print('Mean R2:', test_mean_r2)
        print('Std R2:', test_std_r2)
        print()

print("LEARNING RATE + MAX DEPTH RESULTS")
print("-----------------TRAINING RESULTS--------------------")
print("Default Mean R2 on training:", default_training_r2.mean())
print("Default Mean MSE on training:", default_training_mse.mean())
print("Best Mean R2 on training: ", training_best_R2_using_BOTH," | Max_Depth: ",training_best_MD_R2," | Learning_Rate: ",training_best_LR_R2)
print("Worst Mean R2 on training: ", training_worst_R2_using_BOTH," | Max_Depth: ",training_worst_MD_R2," | Learning_Rate: ",training_worst_LR_R2)
print("Best Mean MSE on training: ", training_best_MSE_using_BOTH," | Max_Depth: ",training_best_MD_MSE," | Learning_Rate: ",training_best_LR_MSE)
print("Worst Mean MSE on training: ", training_worst_MSE_using_BOTH," | Max_Depth: ",training_worst_MD_MSE," | Learning_Rate: ",training_worst_LR_MSE)
print()

print("-----------------TEST RESULTS--------------------")
print("Default Mean R2 on test: ", default_test_r2.mean())
print("Default Mean MSE on test: ", default_test_mse.mean())
print("Best Mean R2 on test: ", test_best_R2_using_BOTH," | Max_Depth: ",test_best_MD_R2," | Learning_Rate: ",test_best_LR_R2)
print("Worst Mean R2 on test: ", test_worst_R2_using_BOTH," | Max_Depth: ",test_worst_MD_R2," | Learning_Rate: ",test_worst_LR_R2)
print("Best Mean MSE on test: ", test_best_MSE_using_BOTH," | Max_Depth: ",test_best_MD_MSE," | Learning_Rate: ",test_best_LR_MSE)
print("Worst Mean MSE on test: ", test_worst_MSE_using_BOTH," | Max_Depth: ",test_worst_MD_MSE," | Learning_Rate: ",test_worst_LR_MSE)








