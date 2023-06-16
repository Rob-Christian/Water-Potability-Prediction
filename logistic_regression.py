## Logistic Regression

# For F1 score list
LR_train_F1 = []
LR_val_F1 = []
LR_test_F1 = []
LR_mean_train_F1 = []
LR_mean_val_F1 = []
LR_mean_test_F1 = []
std_mean_test_F1 = []

# For AUC-ROC score list
LR_test_AUC = []
LR_mean_test_AUC = []
std_mean_test_AUC = []

# For Sensitivity score list
LR_test_sensitivity = []
LR_mean_test_sensitivity = []
std_mean_test_sensitivity = []

# For Specificity score list
LR_test_specificity = []
LR_mean_test_specificity = []
std_mean_test_specificity = []

# For PPV score list
LR_test_PPV = []
LR_mean_test_PPV = []
std_mean_test_PPV = []

# For NPV score list
LR_test_NPV = []
LR_mean_test_NPV = []
std_mean_test_NPV = []

# Hyperparameter consideration for LR
C = [1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]

# Applying grid search for hyperparameter optimization
for c in C:
    lr = LogisticRegression(C = c)
    for train_index, val_index in skf.split(X_subset, y_subset):
        x_train_fold, x_val_fold = X_subset.iloc[train_index], X_subset.iloc[val_index]
        y_train_fold, y_val_fold = y_subset[train_index], y_subset[val_index]
        X_test_fold = norm(X_test, x_train_fold)
        x_val_fold = norm(x_val_fold, x_train_fold)
        x_train_fold = stats.zscore(x_train_fold)
        lr.fit(x_train_fold, y_train_fold)
        tn, fp, fn, tp = confusion_matrix(lr.predict(X_test_fold), y_test).ravel()
        LR_train_F1.append(f1_score(y_train_fold, lr.predict(x_train_fold), average = 'macro'))
        LR_val_F1.append(f1_score(y_val_fold, lr.predict(x_val_fold), average = 'macro'))
        LR_test_F1.append(f1_score(y_test, lr.predict(X_test_fold), average = 'macro'))
        LR_test_AUC.append(roc_auc_score(y_test, lr.predict(X_test_fold)))
        LR_test_sensitivity.append(tp/(tp + fn))
        LR_test_specificity.append(tn/(tn + fp))
        LR_test_PPV.append(tp/(tp + fp))
        LR_test_NPV.append(tn/(tn + fn))
    
    # Updates the mean metrics for each fold
    LR_mean_train_F1.append(round(mean(LR_train_F1), 5))
    LR_mean_val_F1.append(round(mean(LR_val_F1), 5))
    LR_mean_test_F1.append(round(mean(LR_test_F1), 5))
    LR_mean_test_AUC.append(round(mean(LR_test_AUC), 5))
    LR_mean_test_sensitivity.append(round(mean(LR_test_sensitivity), 5))
    LR_mean_test_specificity.append(round(mean(LR_test_specificity), 5))
    LR_mean_test_PPV.append(round(mean(LR_test_PPV), 5))
    LR_mean_test_NPV.append(round(mean(LR_test_NPV), 5))
    std_mean_test_F1.append(round(stdev(LR_test_F1), 5))
    std_mean_test_AUC.append(round(stdev(LR_test_AUC), 5))
    std_mean_test_sensitivity.append(round(stdev(LR_test_sensitivity), 5))
    std_mean_test_specificity.append(round(stdev(LR_test_specificity), 5))
    std_mean_test_PPV.append(round(stdev(LR_test_PPV), 5))
    std_mean_test_NPV.append(round(stdev(LR_test_NPV), 5))
    
    # Clears the list after one fold
    LR_train_F1.clear()
    LR_val_F1.clear()
    LR_test_F1.clear()
    LR_test_AUC.clear()
    LR_test_sensitivity.clear()
    LR_test_specificity.clear()
    LR_test_PPV.clear()
    LR_test_NPV.clear()

# Printing of performance metrics
print('Train F1 Score: ', LR_mean_train_F1)
print('Val F1 Score: ', LR_mean_val_F1)
print('Test F1 Score (mean): ', LR_mean_test_F1)
print('Test F1 Score (std): ', std_mean_test_F1)
print('Test AUC Score (mean): ', LR_mean_test_AUC)
print('Test AUC Score (std): ', std_mean_test_AUC)
print('Test Sensitivity Score (mean): ', LR_mean_test_sensitivity)
print('Test Sensitivity Score (std): ', std_mean_test_sensitivity)
print('Test Specificity Score (mean): ', LR_mean_test_specificity)
print('Test Specificity Score (std): ', std_mean_test_specificity)
print('Test PPV Score (mean): ', LR_mean_test_PPV)
print('Test PPV Score (std): ', std_mean_test_PPV)
print('Test NPV Score (mean): ', LR_mean_test_NPV)
print('Test NPV Score (std): ', std_mean_test_NPV)
