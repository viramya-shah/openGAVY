import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib.pyplot import plot as plt

class metric():
    def __init__(self):
        print ("Metric module")

    def gavy_metric_ks(self, soft, target):
        y_pred = pd.DataFrame(soft, columns = ['score'])
        y_pred['target'] = target
        k_1 = []
        k_2 = []
        for i in range(len(y_pred.index)):
            if y_pred['target'][i] == 0:
                k_1.append(y_pred['score'][i])
            else:
                k_2.append(y_pred['score'][i])    

        KS, p_value = stats.ks_2samp(k_1, k_2)
        print('KS: {}'.format(KS))
        print('P value: {}'.format(p_value))

    def gavy_metric_auc(self, soft, target):
        print("AUC: {}".format(roc_auc_score(target, soft, average = 'micro')))
        fpr, tpr, thres = roc_curve(target, soft)
        plt.plot(fpr, tpr)
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate[1 - Specificity]')
        plt.ylabel('True Positive Rate[Sensitivity]')
        plt.grid(True) 
        plt.show()

    def gavy_corelation(self, df):
        X = df.copy()
        corelation = pd.DataFrame(stats.spearmanr(X)[0])
        corelation.columns = X.columns
        corelation.index = X.columns
        return corelation

    def gavy_crosstable(self, df, target):
        X = df.copy()
        for col_name in X.columns:
            if col_name != target:
                print (pd.crosstab(X.loc[:, target], X.loc[:, col_name], margins = True, normalize = True ))

    # def