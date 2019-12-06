import pandas as pd
import numpy as np
import random

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.preprocessing import LabelEncoder

from matplotlib.pyplot import plot as plt

from scipy import stats

# from metric import *

# from metric import metric_ks, metric_auc