
import numpy as np
import pandas as pd
import statistics
from statistics import mean, stdev
from sklearn.decomposition import PCA
import sklearn
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, make_scorer, recall_score, accuracy_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras import regularizers
from keras.models import load_model
from sklearn.manifold import TSNE

## Classical Machine Learning tools

from statistics import mean, stdev
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

## Feature Engineering

# Delete rows with NaN values
X_raw = pd.read_csv(r"C:\Users\rmcad\Downloads\Research\Water Potability\water_potability.csv")
X_whole = X_raw.dropna(axis = 0)
X_retrieved = X_raw.dropna(axis = 0)
X_retrieved.reset_index(drop = True, inplace = True)

# Good range for PH values
IQR_ph = np.percentile(X_retrieved['ph'], 75) - np.percentile(X_retrieved['ph'], 25)
upper_ph = np.percentile(X_retrieved['ph'], 75) + 1.5*IQR_ph
lower_ph = np.percentile(X_retrieved['ph'], 25) - 1.5*IQR_ph

# Good range for Hardness values
IQR_hard = np.percentile(X_retrieved['Hardness'], 75) - np.percentile(X_retrieved['Hardness'], 25)
upper_hard = np.percentile(X_retrieved['Hardness'], 75) + 1.5*IQR_hard
lower_hard = np.percentile(X_retrieved['Hardness'], 25) - 1.5*IQR_hard

# Good range for Solids values
IQR_solids = np.percentile(X_retrieved['Solids'], 75) - np.percentile(X_retrieved['Solids'], 25)
upper_solids = np.percentile(X_retrieved['Solids'], 75) + 1.5*IQR_solids
lower_solids = np.percentile(X_retrieved['Solids'], 25) - 1.5*IQR_solids

# Good range for Sulfate values
IQR_sulfate = np.percentile(X_retrieved['Sulfate'], 75) - np.percentile(X_retrieved['Sulfate'], 25)
upper_sulfate = np.percentile(X_retrieved['Sulfate'], 75) + 1.5*IQR_sulfate
lower_sulfate = np.percentile(X_retrieved['Sulfate'], 25) - 1.5*IQR_sulfate

# Good range for Conductivity values
IQR_cond = np.percentile(X_retrieved['Conductivity'], 75) - np.percentile(X_retrieved['Conductivity'], 25)
upper_cond = np.percentile(X_retrieved['Conductivity'], 75) + 1.5*IQR_cond
lower_cond = np.percentile(X_retrieved['Conductivity'], 25) - 1.5*IQR_cond

# Good range for Organic Carbon values
IQR_org = np.percentile(X_retrieved['Organic_carbon'], 75) - np.percentile(X_retrieved['Organic_carbon'], 25)
upper_org = np.percentile(X_retrieved['Organic_carbon'], 75) + 1.5*IQR_org
lower_org = np.percentile(X_retrieved['Organic_carbon'], 25) - 1.5*IQR_org

# Good range for Trihalomethanes values
IQR_tri = np.percentile(X_retrieved['Trihalomethanes'], 75) - np.percentile(X_retrieved['Trihalomethanes'], 25)
upper_tri = np.percentile(X_retrieved['Trihalomethanes'], 75) + 1.5*IQR_tri
lower_tri = np.percentile(X_retrieved['Trihalomethanes'], 25) - 1.5*IQR_tri

# Good range for Turbidity values
IQR_tb = np.percentile(X_retrieved['Turbidity'], 75) - np.percentile(X_retrieved['Turbidity'], 25)
upper_tb = np.percentile(X_retrieved['Turbidity'], 75) + 1.5*IQR_tb
lower_tb = np.percentile(X_retrieved['Turbidity'], 25) - 1.5*IQR_tb

X = X_retrieved[(X_retrieved['ph'] >= lower_ph) & (X_retrieved['ph'] <= upper_ph) &
               (X_retrieved['Hardness'] >= lower_hard) & (X_retrieved['Hardness'] <= upper_hard) &
            (X_retrieved['Solids'] >= lower_solids) & (X_retrieved['Solids'] <= upper_solids) &
            (X_retrieved['Sulfate'] >= lower_sulfate) & (X_retrieved['Sulfate'] <= upper_sulfate) &
            (X_retrieved['Conductivity'] >= lower_cond) & (X_retrieved['Conductivity'] <= upper_cond) &
            (X_retrieved['Organic_carbon'] >= lower_org) & (X_retrieved['Organic_carbon'] <= upper_org) &
            (X_retrieved['Trihalomethanes'] >= lower_tri) & (X_retrieved['Trihalomethanes'] <= upper_tri) &
            (X_retrieved['Turbidity'] >= lower_tb) & (X_retrieved['Turbidity'] <= upper_tb)]

X_retrieved = X_retrieved[(X_retrieved['ph'] >= lower_ph) & (X_retrieved['ph'] <= upper_ph) &
               (X_retrieved['Hardness'] >= lower_hard) & (X_retrieved['Hardness'] <= upper_hard) &
            (X_retrieved['Solids'] >= lower_solids) & (X_retrieved['Solids'] <= upper_solids) &
            (X_retrieved['Sulfate'] >= lower_sulfate) & (X_retrieved['Sulfate'] <= upper_sulfate) &
            (X_retrieved['Conductivity'] >= lower_cond) & (X_retrieved['Conductivity'] <= upper_cond) &
            (X_retrieved['Organic_carbon'] >= lower_org) & (X_retrieved['Organic_carbon'] <= upper_org) &
            (X_retrieved['Trihalomethanes'] >= lower_tri) & (X_retrieved['Trihalomethanes'] <= upper_tri) &
            (X_retrieved['Turbidity'] >= lower_tb) & (X_retrieved['Turbidity'] <= upper_tb)]

# Drop potability in X
y = X['Potability'].to_numpy()
y = np.reshape(y, (X.shape[0], 1))
X.drop("Potability", axis = 1, inplace = True)

## Use Principal Component Analysis

X_PCA = stats.zscore(X)

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(X_PCA)

# Store the principal component values in a dataframe

principal_df = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])

# Print the total variance (percentage remain after dimension reduction)

print("The total variance is: " + str(pca.explained_variance_ratio_.sum() * 100))

# Plot the reduced dimension with colored labels

plt.figure(figsize=(8,8))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('PC1: ' + str(round(100*pca.explained_variance_ratio_[0], 2)) + '%',fontsize=20)
plt.ylabel('PC2: ' + str(round(100*pca.explained_variance_ratio_[1], 2)) + '%',fontsize=20)
plt.title("Principal Component Analysis of Water Potability",fontsize=20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = y == target
    plt.scatter(principal_df.loc[indicesToKeep, 'PC1']
               , principal_df.loc[indicesToKeep, 'PC2'], c = color, s = 50)

plt.legend(['Non Potable', 'Potable'],prop={'size': 15})

## Use t-Distributed Stochastic Neighbor Embedding with 2 components only

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X_PCA)

df_subset = pd.DataFrame()

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(8,8))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.title("t-SNE of Water Potability Dataset",fontsize=20)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = y == target
    plt.scatter(df_subset.loc[indicesToKeep, 'tsne-2d-one']
               , df_subset.loc[indicesToKeep, 'tsne-2d-two'], c = color, s = 50)

plt.legend(['Not Potable', 'Potable'],prop={'size': 15})

# Normalization Process

def norm(test, train):
    mean_trans = train.mean().to_numpy().reshape((train.shape[1],1))*np.ones((1,test.shape[0]))
    mean = mean_trans.T
    std_trans = train.std().to_numpy().reshape((train.shape[1],1))*np.ones((1,test.shape[0]))
    std = std_trans.T
    transformed = np.divide(test.to_numpy() - mean, std)
    return pd.DataFrame(transformed, columns = train.columns)

X_subset, X_test, y_subset, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42, shuffle = True, stratify = y)
skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

## Logistic Regression

# For F1 score
LR_train_F1 = []
LR_val_F1 = []
LR_test_F1 = []
LR_mean_train_F1 = []
LR_mean_val_F1 = []
LR_mean_test_F1 = []
std_mean_test_F1 = []

# For AUC-ROC score
LR_test_AUC = []
LR_mean_test_AUC = []
std_mean_test_AUC = []

# For Sensitivity score
LR_test_sensitivity = []
LR_mean_test_sensitivity = []
std_mean_test_sensitivity = []

# For Specificity score
LR_test_specificity = []
LR_mean_test_specificity = []
std_mean_test_specificity = []

# For PPV score
LR_test_PPV = []
LR_mean_test_PPV = []
std_mean_test_PPV = []

# For NPV score
LR_test_NPV = []
LR_mean_test_NPV = []
std_mean_test_NPV = []

C = [1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]

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
    LR_train_F1.clear()
    LR_val_F1.clear()
    LR_test_F1.clear()
    LR_test_AUC.clear()
    LR_test_sensitivity.clear()
    LR_test_specificity.clear()
    LR_test_PPV.clear()
    LR_test_NPV.clear()

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

## Support Vector Machines

# For F1 score
SV_train_F1 = []
SV_val_F1 = []
SV_test_F1 = []
SV_mean_train_F1 = []
SV_mean_val_F1 = []
SV_mean_test_F1 = []
std_mean_test_F1 = []

# For AUC-ROC score
SV_test_AUC = []
SV_mean_test_AUC = []
std_mean_test_AUC = []

# For Sensitivity score
SV_test_sensitivity = []
SV_mean_test_sensitivity = []
std_mean_test_sensitivity = []

# For Specificity score
SV_test_specificity = []
SV_mean_test_specificity = []
std_mean_test_specificity = []

# For PPV score
SV_test_PPV = []
SV_mean_test_PPV = []
std_mean_test_PPV = []

# For NPV score
SV_test_NPV = []
SV_mean_test_NPV = []
std_mean_test_NPV = []

C = [1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]

for c in C:
    svc = SVC(kernel = 'rbf', C = c, probability = True)
    for train_index, val_index in skf.split(X_subset, y_subset):
        x_train_fold, x_val_fold = X_subset.iloc[train_index], X_subset.iloc[val_index]
        y_train_fold, y_val_fold = y_subset[train_index], y_subset[val_index]
        X_test_fold = norm(X_test, x_train_fold)
        x_val_fold = norm(x_val_fold, x_train_fold)
        x_train_fold = stats.zscore(x_train_fold)
        svc.fit(x_train_fold, y_train_fold)
        tn, fp, fn, tp = confusion_matrix(svc.predict(X_test_fold), y_test).ravel()
        SV_train_F1.append(f1_score(y_train_fold, svc.predict(x_train_fold), average = 'macro'))
        SV_val_F1.append(f1_score(y_val_fold, svc.predict(x_val_fold), average = 'macro'))
        SV_test_F1.append(f1_score(y_test, svc.predict(X_test_fold), average = 'macro'))
        SV_test_AUC.append(roc_auc_score(y_test, svc.predict(X_test_fold)))
        SV_test_sensitivity.append(tp/(tp + fn))
        SV_test_specificity.append(tn/(tn + fp))
        SV_test_PPV.append(tp/(tp + fp))
        SV_test_NPV.append(tn/(tn + fn))
    SV_mean_train_F1.append(round(mean(SV_train_F1), 5))
    SV_mean_val_F1.append(round(mean(SV_val_F1), 5))
    SV_mean_test_F1.append(round(mean(SV_test_F1), 5))
    SV_mean_test_AUC.append(round(mean(SV_test_AUC), 5))
    SV_mean_test_sensitivity.append(round(mean(SV_test_sensitivity), 5))
    SV_mean_test_specificity.append(round(mean(SV_test_specificity), 5))
    SV_mean_test_PPV.append(round(mean(SV_test_PPV), 5))
    SV_mean_test_NPV.append(round(mean(SV_test_NPV), 5))
    std_mean_test_F1.append(round(stdev(SV_test_F1), 5))
    std_mean_test_AUC.append(round(stdev(SV_test_AUC), 5))
    std_mean_test_sensitivity.append(round(stdev(SV_test_sensitivity), 5))
    std_mean_test_specificity.append(round(stdev(SV_test_specificity), 5))
    std_mean_test_PPV.append(round(stdev(SV_test_PPV), 5))
    std_mean_test_NPV.append(round(stdev(SV_test_NPV), 5))
    SV_train_F1.clear()
    SV_val_F1.clear()
    SV_test_F1.clear()
    SV_test_AUC.clear()
    SV_test_sensitivity.clear()
    SV_test_specificity.clear()
    SV_test_PPV.clear()
    SV_test_NPV.clear()

print('Train F1 Score: ', SV_mean_train_F1)
print('Val F1 Score: ', SV_mean_val_F1)
print('Test F1 Score (mean): ', SV_mean_test_F1)
print('Test F1 Score (std): ', std_mean_test_F1)
print('Test AUC Score (mean): ', SV_mean_test_AUC)
print('Test AUC Score (std): ', std_mean_test_AUC)
print('Test Sensitivity Score (mean): ', SV_mean_test_sensitivity)
print('Test Sensitivity Score (std): ', std_mean_test_sensitivity)
print('Test Specificity Score (mean): ', SV_mean_test_specificity)
print('Test Specificity Score (std): ', std_mean_test_specificity)
print('Test PPV Score (mean): ', SV_mean_test_PPV)
print('Test PPV Score (std): ', std_mean_test_PPV)
print('Test NPV Score (mean): ', SV_mean_test_NPV)
print('Test NPV Score (std): ', std_mean_test_NPV)

## Random Forest

# For F1 score
RF_train_F1 = []
RF_val_F1 = []
RF_test_F1 = []
RF_mean_train_F1 = []
RF_mean_val_F1 = []
RF_mean_test_F1 = []
std_mean_test_F1 = []

# For AUC-ROC score
RF_test_AUC = []
RF_mean_test_AUC = []
std_mean_test_AUC = []

# For Sensitivity score
RF_test_sensitivity = []
RF_mean_test_sensitivity = []
std_mean_test_sensitivity = []

# For Specificity score
RF_test_specificity = []
RF_mean_test_specificity = []
std_mean_test_specificity = []

# For PPV score
RF_test_PPV = []
RF_mean_test_PPV = []
std_mean_test_PPV = []

# For NPV score
RF_test_NPV = []
RF_mean_test_NPV = []
std_mean_test_NPV = []

n_estimators = [100, 150, 200, 250, 300]

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
    RF_train_F1.clear()
    RF_val_F1.clear()
    RF_test_F1.clear()
    RF_test_AUC.clear()
    RF_test_sensitivity.clear()
    RF_test_specificity.clear()
    RF_test_PPV.clear()
    RF_test_NPV.clear()

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

## Artificial Neural Networks (2 layers)

# For F1 score
NN2_train_F1 = []
NN2_val_F1 = []
NN2_test_F1 = []
NN2_mean_train_F1 = []
NN2_mean_val_F1 = []
NN2_mean_test_F1 = []
std_mean_test_F1 = []

# For AUC-ROC score
NN2_test_AUC = []
NN2_mean_test_AUC = []
std_mean_test_AUC = []

# For Sensitivity score
NN2_test_sensitivity = []
NN2_mean_test_sensitivity = []
std_mean_test_sensitivity = []

# For Specificity score
NN2_test_specificity = []
NN2_mean_test_specificity = []
std_mean_test_specificity = []

# For PPV score
NN2_test_PPV = []
NN2_mean_test_PPV = []
std_mean_test_PPV = []

# For NPV score
NN2_test_NPV = []
NN2_mean_test_NPV = []
std_mean_test_NPV = []

j = 0
i_s = 0
l = [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

for lc in l:
    model = keras.Sequential([
    keras.layers.Dense(50, input_shape=(9,), activation = 'relu', kernel_regularizer = regularizers.l2(lc)),
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
        NN2_train_F1.append(f1_score(y_train_fold, y_train_pred, average = 'macro'))
        NN2_val_F1.append(f1_score(y_val_fold, y_val_pred, average = 'macro'))
        NN2_test_F1.append(f1_score(y_test, y_test_pred, average = 'macro'))
        NN2_test_AUC.append(roc_auc_score(y_test, y_test_pred))
        NN2_test_sensitivity.append(tp/(tp + fn))
        NN2_test_specificity.append(tn/(tn + fp))
        NN2_test_PPV.append(tp/(tp + fp))
        NN2_test_NPV.append(tn/(tn + fn))
        model.save(r'C:\Users\rmcad\Downloads\Research\Water Potability\Architecture1\a{}.h5'.format(i_s+1))
        i_s += 1
    NN2_mean_train_F1.append(round(mean(NN2_train_F1), 5))
    NN2_mean_val_F1.append(round(mean(NN2_val_F1), 5))
    NN2_mean_test_F1.append(round(mean(NN2_test_F1), 5))
    NN2_mean_test_AUC.append(round(mean(NN2_test_AUC), 5))
    NN2_mean_test_sensitivity.append(round(mean(NN2_test_sensitivity), 5))
    NN2_mean_test_specificity.append(round(mean(NN2_test_specificity), 5))
    NN2_mean_test_PPV.append(round(mean(NN2_test_PPV), 5))
    NN2_mean_test_NPV.append(round(mean(NN2_test_NPV), 5))
    std_mean_test_F1.append(round(stdev(NN2_test_F1), 5))
    std_mean_test_AUC.append(round(stdev(NN2_test_AUC), 5))
    std_mean_test_sensitivity.append(round(stdev(NN2_test_sensitivity), 5))
    std_mean_test_specificity.append(round(stdev(NN2_test_specificity), 5))
    std_mean_test_PPV.append(round(stdev(NN2_test_PPV), 5))
    std_mean_test_NPV.append(round(stdev(NN2_test_NPV), 5))
    NN2_train_F1.clear()
    NN2_val_F1.clear()
    NN2_test_F1.clear()
    NN2_test_AUC.clear()
    NN2_test_sensitivity.clear()
    NN2_test_specificity.clear()
    NN2_test_PPV.clear()
    NN2_test_NPV.clear()

print('Train F1 Score: ', NN2_mean_train_F1)
print('Val F1 Score: ', NN2_mean_val_F1)
print('Test F1 Score (mean): ', NN2_mean_test_F1)
print('Test F1 Score (std): ', std_mean_test_F1)
print('Test AUC Score (mean): ', NN2_mean_test_AUC)
print('Test AUC Score (std): ', std_mean_test_AUC)
print('Test Sensitivity Score (mean): ', NN2_mean_test_sensitivity)
print('Test Sensitivity Score (std): ', std_mean_test_sensitivity)
print('Test Specificity Score (mean): ', NN2_mean_test_specificity)
print('Test Specificity Score (std): ', std_mean_test_specificity)
print('Test PPV Score (mean): ', NN2_mean_test_PPV)
print('Test PPV Score (std): ', std_mean_test_PPV)
print('Test NPV Score (mean): ', NN2_mean_test_NPV)
print('Test NPV Score (std): ', std_mean_test_NPV)

## Artificial Neural Networks (4 layers)

# For F1 score
NN4_train_F1 = []
NN4_val_F1 = []
NN4_test_F1 = []
NN4_mean_train_F1 = []
NN4_mean_val_F1 = []
NN4_mean_test_F1 = []
std_mean_test_F1 = []

# For AUC-ROC score
NN4_test_AUC = []
NN4_mean_test_AUC = []
std_mean_test_AUC = []

# For Sensitivity score
NN4_test_sensitivity = []
NN4_mean_test_sensitivity = []
std_mean_test_sensitivity = []

# For Specificity score
NN4_test_specificity = []
NN4_mean_test_specificity = []
std_mean_test_specificity = []

# For PPV score
NN4_test_PPV = []
NN4_mean_test_PPV = []
std_mean_test_PPV = []

# For NPV score
NN4_test_NPV = []
NN4_mean_test_NPV = []
std_mean_test_NPV = []

j = 0
i_s = 0
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
        model.save(r'C:\Users\rmcad\Downloads\Research\Water Potability\Architecture3\a{}.h5'.format(i_s+1))
        i_s += 1
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
    NN4_train_F1.clear()
    NN4_val_F1.clear()
    NN4_test_F1.clear()
    NN4_test_AUC.clear()
    NN4_test_sensitivity.clear()
    NN4_test_specificity.clear()
    NN4_test_PPV.clear()
    NN4_test_NPV.clear()

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

def affine_backward(dout, cache):
    x, w = cache
    dx = np.dot(dout, w.T)
    return dx

def relu_backward(dout, cache):
    x = cache
    dx = dout * np.where(x > 0, np.ones(x.shape), np.zeros(x.shape))
    return dx

def sigmoid_backward(cache):
    dx = cache*(1 - cache)
    return dx

## Sensitivity and Probability of Artificial Neural Networks (2 layers)

X = stats.zscore(X)

NN2_sensitivity_fold = np.zeros((1, 9, X.shape[0]))
NN2_sensitivity = np.zeros((1, 9, 10))
NN2_mean_sensitivity = np.zeros((1, 9))

for i1 in range(10):
    saved_model = load_model(r'C:\Users\rmcad\Downloads\Research\Water Potability\Architecture1\a{}.h5'.format(10*3+i1+1))
    model2 = keras.Sequential([keras.layers.Dense(50, input_shape=(9,), weights=saved_model.layers[0].get_weights(), activation = 'relu')])
    for i2 in range(X.shape[0]):
        dout = sigmoid_backward(saved_model.predict(X.iloc[i2:i2+1]))
        dout = affine_backward(dout, (X.iloc[i2:i2+1], saved_model.get_weights()[2]))
        dout = relu_backward(dout, model2.predict(X.iloc[i2:i2+1]))
        dout = affine_backward(dout, (X.iloc[i2:i2+1], saved_model.get_weights()[0]))
        NN2_sensitivity_fold[:,:,i2] = dout
    NN2_sensitivity[:,:,i1] = np.mean(NN2_sensitivity_fold, axis = 2)
    NN2_sensitivity_fold = np.zeros((1, 9, X.shape[0]))
NN2_mean_sensitivity = np.mean(NN2_sensitivity, axis = 2)
df1 = pd.DataFrame(NN2_mean_sensitivity.T)
df1.to_excel(r'C:\Users\rmcad\Downloads\Research\Water Potability\Binary Sensitivities ANN2.xlsx', index = False)



