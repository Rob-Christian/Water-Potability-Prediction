## Artificial Neural Networks (4 layers)

# For F1 score list
NN4_train_F1 = []
NN4_val_F1 = []
NN4_test_F1 = []
NN4_mean_train_F1 = []
NN4_mean_val_F1 = []
NN4_mean_test_F1 = []
std_mean_test_F1 = []

# For AUC-ROC score list
NN4_test_AUC = []
NN4_mean_test_AUC = []
std_mean_test_AUC = []

# For Sensitivity score list
NN4_test_sensitivity = []
NN4_mean_test_sensitivity = []
std_mean_test_sensitivity = []

# For Specificity score list
NN4_test_specificity = []
NN4_mean_test_specificity = []
std_mean_test_specificity = []

# For PPV score list
NN4_test_PPV = []
NN4_mean_test_PPV = []
std_mean_test_PPV = []

# For NPV score list
NN4_test_NPV = []
NN4_mean_test_NPV = []
std_mean_test_NPV = []

# For saving of weighted architectures
j = 0
i_s = 0

# Hyperparameter consideration for ANN4
l = [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

for lc in l:
    model = keras.Sequential([
    keras.layers.Dense(50, input_shape=(9,), activation = 'relu', kernel_regularizer = regularizers.l2(lc)),
    keras.layers.Dense(50, activation = 'relu', kernel_regularizer = regularizers.l2(lc)),
    keras.layers.Dense(50, activation = 'relu', kernel_regularizer = regularizers.l2(lc)),
    keras.layers.Dense(1, activation = 'sigmoid')
    ])
    model.compile(optimizer = 'adam',
            loss = 'binary_crossentropy',
            metrics = ['accuracy'])
    for train_index, val_index in skf.split(X_subset, y_subset):
        x_train_fold, x_val_fold = X_subset.iloc[train_index], X_subset.iloc[val_index]
        y_train_fold, y_val_fold = y_subset[train_index], y_subset[val_index]
        X_test_fold = norm(X_test, x_train_fold)
        x_val_fold = norm(x_val_fold, x_train_fold)
        x_train_fold = stats.zscore(x_train_fold)
        model.fit(x_train_fold, y_train_fold, epochs = 200, batch_size = len(x_train_fold), verbose = 0)
        y_train_pred = np.where(model.predict(x_train_fold) < 0.5, 0, 1)
        y_val_pred = np.where(model.predict(x_val_fold) < 0.5, 0, 1)
        y_test_pred = np.where(model.predict(X_test_fold) < 0.5, 0, 1)
        tn, fp, fn, tp = confusion_matrix(y_test_pred, y_test).ravel()
        NN4_train_F1.append(f1_score(y_train_fold, y_train_pred, average = 'macro'))
        NN4_val_F1.append(f1_score(y_val_fold, y_val_pred, average = 'macro'))
        NN4_test_F1.append(f1_score(y_test, y_test_pred, average = 'macro'))
        NN4_test_AUC.append(roc_auc_score(y_test, y_test_pred))
        NN4_test_sensitivity.append(tp/(tp + fn))
        NN4_test_specificity.append(tn/(tn + fp))
        NN4_test_PPV.append(tp/(tp + fp))
        NN4_test_NPV.append(tn/(tn + fn))
        
        # Saves model architecture
        model.save(r'C:\Users\rmcad\Downloads\Research\Water Potability\Architecture3\a{}.h5'.format(i_s+1))
        i_s += 1
    
    # Updates the mean metrics for each fold
    NN4_mean_train_F1.append(round(mean(NN4_train_F1), 5))
    NN4_mean_val_F1.append(round(mean(NN4_val_F1), 5))
    NN4_mean_test_F1.append(round(mean(NN4_test_F1), 5))
    NN4_mean_test_AUC.append(round(mean(NN4_test_AUC), 5))
    NN4_mean_test_sensitivity.append(round(mean(NN4_test_sensitivity), 5))
    NN4_mean_test_specificity.append(round(mean(NN4_test_specificity), 5))
    NN4_mean_test_PPV.append(round(mean(NN4_test_PPV), 5))
    NN4_mean_test_NPV.append(round(mean(NN4_test_NPV), 5))
    std_mean_test_F1.append(round(stdev(NN4_test_F1), 5))
    std_mean_test_AUC.append(round(stdev(NN4_test_AUC), 5))
    std_mean_test_sensitivity.append(round(stdev(NN4_test_sensitivity), 5))
    std_mean_test_specificity.append(round(stdev(NN4_test_specificity), 5))
    std_mean_test_PPV.append(round(stdev(NN4_test_PPV), 5))
    std_mean_test_NPV.append(round(stdev(NN4_test_NPV), 5))
    
    # Clears the list after one fold
    NN4_train_F1.clear()
    NN4_val_F1.clear()
    NN4_test_F1.clear()
    NN4_test_AUC.clear()
    NN4_test_sensitivity.clear()
    NN4_test_specificity.clear()
    NN4_test_PPV.clear()
    NN4_test_NPV.clear()

# Printing of performance metrics
print('Train F1 Score: ', NN4_mean_train_F1)
print('Val F1 Score: ', NN4_mean_val_F1)
print('Test F1 Score (mean): ', NN4_mean_test_F1)
print('Test F1 Score (std): ', std_mean_test_F1)
print('Test AUC Score (mean): ', NN4_mean_test_AUC)
print('Test AUC Score (std): ', std_mean_test_AUC)
print('Test Sensitivity Score (mean): ', NN4_mean_test_sensitivity)
print('Test Sensitivity Score (std): ', std_mean_test_sensitivity)
print('Test Specificity Score (mean): ', NN4_mean_test_specificity)
print('Test Specificity Score (std): ', std_mean_test_specificity)
print('Test PPV Score (mean): ', NN4_mean_test_PPV)
print('Test PPV Score (std): ', std_mean_test_PPV)
print('Test NPV Score (mean): ', NN4_mean_test_NPV)
print('Test NPV Score (std): ', std_mean_test_NPV)
