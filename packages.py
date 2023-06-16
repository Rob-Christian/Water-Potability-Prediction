## General Packages

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
import scipy.stats as stats
import matplotlib.pyplot as plt
import statistics
from statistics import mean, stdev

## Specific parts from tensorflow

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras import regularizers
from keras.models import load_model

## Specific parts from scikit learn

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, make_scorer, recall_score, accuracy_score, precision_score, f1_score, confusion_matrix

## Classical Machine Learning tools

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
