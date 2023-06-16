## Random Forest

# For F1 score list
RF_train_F1 = []
RF_val_F1 = []
RF_test_F1 = []
RF_mean_train_F1 = []
RF_mean_val_F1 = []
RF_mean_test_F1 = []
std_mean_test_F1 = []

# For AUC-ROC score list
RF_test_AUC = []
RF_mean_test_AUC = []
std_mean_test_AUC = []

# For Sensitivity score list
RF_test_sensitivity = []
RF_mean_test_sensitivity = []
std_mean_test_sensitivity = []

# For Specificity score list
RF_test_specificity = []
RF_mean_test_specificity = []
std_mean_test_specificity = []

# For PPV score list
RF_test_PPV = []
RF_mean_test_PPV = []
std_mean_test_PPV = []

# For NPV score list
RF_test_NPV = []
RF_mean_test_NPV = []
std_mean_test_NPV = []

# Hyperparameter consideration for RF
n_estimators = [100, 150, 200, 250, 300]

# Applying grid search for hyperparameter optimization
for n in n_estimators:
    rf = RandomForestClassifier(n_estimators = n, min_samples_leaf = 10)
    for train_index, val_index in skf.split(X_subset, y_subset):
        x_train_fold, x_val_fold = X_subset.iloc[train_index], X_subset.iloc[val_index]
        y_train_fold, y_val_fold = y_subset[train_index], y_subset[val_index]
        X_test_fold = norm(X_test, x_train_fold)
        x_val_fold = norm(x_val_fold, x_train_fold)
        x_train_fold = stats.zscore(x_train_fold)
        rf.fit(x_train_fold, y_train_fold)
        tn, fp, fn, tp = confusion_matrix(rf.predict(X_test_fold), y_test).ravel()
        RF_train_F1.append(f1_score(y_train_fold, rf.predict(x_train_fold), average = 'macro'))
        RF_val_F1.append(f1_score(y_val_fold, rf.predict(x_val_fold), average = 'macro'))
        RF_test_F1.append(f1_score(y_test, rf.predict(X_test_fold), average = 'macro'))
        RF_test_AUC.append(roc_auc_score(y_test, rf.predict(X_test_fold)))
        RF_test_sensitivity.append(tp/(tp + fn))
        RF_test_specificity.append(tn/(tn + fp))
        RF_test_PPV.append(tp/(tp + fp))
        RF_test_NPV.append(tn/(tn + fn))
    
    # Updates the mean metrics for each fold
    RF_mean_train_F1.append(round(mean(RF_train_F1), 5))
    RF_mean_val_F1.append(round(mean(RF_val_F1), 5))
    RF_mean_test_F1.append(round(mean(RF_test_F1), 5))
    RF_mean_test_AUC.append(round(mean(RF_test_AUC), 5))
    RF_mean_test_sensitivity.append(round(mean(RF_test_sensitivity), 5))
    RF_mean_test_specificity.append(round(mean(RF_test_specificity), 5))
    RF_mean_test_PPV.append(round(mean(RF_test_PPV), 5))
    RF_mean_test_NPV.append(round(mean(RF_test_NPV), 5))
    std_mean_test_F1.append(round(stdev(RF_test_F1), 5))
    std_mean_test_AUC.append(round(stdev(RF_test_AUC), 5))
    std_mean_test_sensitivity.append(round(stdev(RF_test_sensitivity), 5))
    std_mean_test_specificity.append(round(stdev(RF_test_specificity), 5))
    std_mean_test_PPV.append(round(stdev(RF_test_PPV), 5))
    std_mean_test_NPV.append(round(stdev(RF_test_NPV), 5))
    
    # Clears the list after one fold
    RF_train_F1.clear()
    RF_val_F1.clear()
    RF_test_F1.clear()
    RF_test_AUC.clear()
    RF_test_sensitivity.clear()
    RF_test_specificity.clear()
    RF_test_PPV.clear()
    RF_test_NPV.clear()

# Printing of performance metrics
print('Train F1 Score: ', RF_mean_train_F1)
print('Val F1 Score: ', RF_mean_val_F1)
print('Test F1 Score (mean): ', RF_mean_test_F1)
print('Test F1 Score (std): ', std_mean_test_F1)
print('Test AUC Score (mean): ', RF_mean_test_AUC)
print('Test AUC Score (std): ', std_mean_test_AUC)
print('Test Sensitivity Score (mean): ', RF_mean_test_sensitivity)
print('Test Sensitivity Score (std): ', std_mean_test_sensitivity)
print('Test Specificity Score (mean): ', RF_mean_test_specificity)
print('Test Specificity Score (std): ', std_mean_test_specificity)
print('Test PPV Score (mean): ', RF_mean_test_PPV)
print('Test PPV Score (std): ', std_mean_test_PPV)
print('Test NPV Score (mean): ', RF_mean_test_NPV)
print('Test NPV Score (std): ', std_mean_test_NPV)
