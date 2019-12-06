import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

# from metric import *
from scipy import stats
from sklearn import metrics
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

def gavy_metric_ks(soft, target):
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

def gavy_metric_auc(soft, target):
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

def model_predictor(alg, df_test, target_var, printKS = 1, printAUCcurve = 1):
    x_test=df_test.loc[:,df_test.columns!=target_var]
    X_names = np.array(x_test.columns)

    df_target = df_test[target_var]
    y_test = np.array(df_target)

    y_test = y_test.reshape(len(y_test),)
    hard_predictions_test = alg.predict(x_test)
    soft_predictions_test = alg.predict_proba(x_test)[:, 1]
    
    print ("\n#######################################")
    print ("\n#############TEST RESULTS##############")
    print ("\n#######################################")
    print ("\nModel Report")
    print ("AUC Score (Test): %f" % roc_auc_score(y_test, soft_predictions_test))
    
    if printKS:
        print ("\n#### KS and p-val on test set####")
        gavy_metric_ks(soft_predictions_test, y_test)
    
    if printAUCcurve:
        print ("\n#### ROC curve (Test set)####")
        gavy_metric_auc(soft_predictions_test, y_test)
    
    return soft_predictions_test


class modelling():
    def __init__(self):
        print ("Welcome to [MODELLING]")
        print("")

    def open_svm(self, df_train, response_var, df_test, dict_paramters = {}, performCV = 1, cv_folds = 10, printKS = 1, printAUCcurve = 1, random_state = 29):
        X = df_train.copy()

        X_train = X.loc[:, X.columns != response_var]
        col_names = X.columns

        y_train = np.array(X.loc[:,response_var])

        if 'C' in dict_paramters:
            c = dict_paramters['C']
        else:
            c = 0.01
        
        if 'kernel' in dict_paramters:
            kernel = dict_paramters['kernel']
        else:
            kernel = 'rbf'

        if 'degree' in dict_paramters:
            degree = dict_paramters['degree']
        else:
            degree = 3
        
        if 'random_State' in dict_paramters:
            random_state = dict_paramters['random_state']
        else:
            random_state = 29

        if 'max_iter' in dict_paramters:
            max_iter = dict_paramters['max_iter']
        else:
            max_iter = 100

        if 'probability' in dict_paramters:
            probability = dict_paramters['probability']
        else:
            probability = False

        clf_open_svm = SVC(C= c, kernel = kernel, degree = degree, probability = probability, max_iter = max_iter, random_state = random_state)
        clf_open_svm.fit(X_train, y_train) 
        
        # prob > 0.5 => 1 else 0
        hard_predictions_train = clf_open_svm.predict(X_train)
        
        # considering only class = 1: either binary or one-vs-all
        soft_predictions_train = clf_open_svm.predict_proba(X_train)[:,1]

        if performCV:
            cv_score = cross_val_score(clf_open_svm, X_train, y_train, cv = cv_folds, scoring = 'roc_auc')
        
        print ("\n###########################################")
        print ("\n#############TRAINING RESULTS##############")
        print ("\n###########################################")
        print ("\nModel Report")

        print ("AUC Score (Train): %f" % roc_auc_score(y_train, soft_predictions_train))
        
        if performCV:
            print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

        if printKS:
            print ("\n#### KS and p-val on Train set####")
            gavy_metric_ks(soft = soft_predictions_train, target = y_train)
        
        if printAUCcurve:
            print ("\n#### ROC curve (Train set)####")
            gavy_metric_auc(soft = soft_predictions_train, target = y_train)
            
        model_predictor(clf_open_svm, df_test, response_var, printKS, printAUCcurve)

        return clf_open_svm, soft_predictions_train, hard_predictions_train

    def open_naive_bayes(self, df_train, response_var, df_test, dict_paramters = {}, performCV = 1, cv_folds = 10, printKS = 1, printAUCcurve = 1):
        
        X = df_train.copy()
        X_train = X.loc[:, X.columns != response_var]
        y_train = np.array(X.loc[:, response_var])
        
        if 'priors' in dict_paramters:
            priors = dict_paramters['priors']
        else:
            priors=None, 
            
        if 'var_smoothing' in dict_paramters:
            var_smoothing = dict_paramters['var_smoothing']
        else:
            var_smoothing = 1e-09

        clf_open_nb = GaussianNB(priors = priors, var_smoothing = var_smoothing)
        clf_open_nb.fit(X_train, y_train)

        # prob > 0.5 => 1 else 0
        hard_predictions_train = clf_open_nb.predict(X_train)
        
        # considering only class = 1: either binary or one-vs-all
        soft_predictions_train = clf_open_nb.predict_proba(X_train)[:,1]

        if performCV:
            cv_score = cross_val_score(clf_open_nb, X_train, y_train, cv = cv_folds, scoring = 'roc_auc')
        
        print ("\n###########################################")
        print ("\n#############TRAINING RESULTS##############")
        print ("\n###########################################")
        print ("\nModel Report")

        print ("AUC Score (Train): %f" % roc_auc_score(y_train, soft_predictions_train))
        
        if performCV:
            print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

        if printKS:
            print ("\n#### KS and p-val on Train set####")
            gavy_metric_ks(soft = soft_predictions_train, target = y_train)
        
        if printAUCcurve:
            print ("\n#### ROC curve (Train set)####")
            gavy_metric_auc(soft = soft_predictions_train, target = y_train)
            
        model_predictor(clf_open_nb, df_test, response_var, printKS, printAUCcurve)
        
        return clf_open_nb, soft_predictions_train, hard_predictions_train   

    def open_regression(self, df_train, df_test, target_var, method = "gradient_descent",alpha = 0.01, iter = 10):
    
        if(method == 'gradient_descent'):    
            train_gd = df_train.drop([target_var],axis = 1)
            train_gd_names = np.array(train_gd.columns)
            train_gd = np.array(train_gd) 
            # Stuffing 1's in the X_train
            X_train = np.c_[np.ones(train_gd.shape),train_gd]

            df_target_train = df_train[target_var]
            Y_train = np.array(df_target_train)


            test_gd = df_test.drop([target_var],axis = 1)
            test_gd_names = np.array(test_gd.columns)
            test_gd = np.array(test_gd)
            # Stuffing 1's in the X_test
            X_test = np.c_[np.ones(test_gd.shape),test_gd]

            df_target_test = df_test[target_var]
            Y_test = np.array(df_target_test)
            
            N = X_train.shape[0]
            features = X_train.shape[1]

            # Initialising theta with random numbers of size (features,1)
            theta = np.zeros(features)
            
            for i in range(0,iter+1):
                hyp = np.matmul(X_train, theta)
                theta[0] = theta[0] - (alpha/N)*sum(hyp - Y_train)
                for j in range(1,features):
                    theta[j] = theta[j] - (alpha/N)*sum((hyp - Y_train) * X_train.transpose()[j])

            y_predict_train = np.matmul(X_train, theta)
            y_predict_test = np.matmul(X_test,theta)
            print("\n##############MODELLING MODE ON#############")
            print ("\n#############PERFORMANCE GRADIENT DESCENT################")
            print ("\n#############TRAINING RESULTS##############")
            print ("\n###########################################")
            print ("\nModel Report")

            mse_train = metrics.mean_squared_error(y_predict_train, Y_train)
            print("Mean Squared ERROR on TRAIN SET:{}".format(mse_train))
            
            print ("\n#############TESTING RESULTS##############")
            mse_test = metrics.mean_squared_error(Y_test,y_predict_test)
            print("Mean Squared ERROR on TEST SET:{}".format(mse_test))

            return theta, mse_train, mse_test

        if(method == 'closed'):

            train_closed = df_train.drop([target_var],axis = 1)
            train_closed_names = np.array(train_closed.columns)
            train_closed = np.array(train_closed) 

            df_target_train = df_train[target_var]
            X_train = np.c_[np.ones(train_closed.shape),train_closed]
            
            y_train = np.array(df_target_train)
            
            print("\n##############MODELLING MODE ON#############")
            print ("\n#############PERFORMANCE CLOSED FORM################")
            print ("\n#############TRAINING RESULTS##############")
            print ("\n###########################################")
            print ("\n##############Model Report###############")
            
            test_closed = df_test.drop([target_var],axis = 1)
            test_closed_names = np.array(test_closed.columns)
            test_closed = np.array(test_closed)
            

            X_test = np.c_[np.ones(test_closed.shape),test_closed]
            df_target_test = df_test[target_var]
            Y_test = np.array(df_target_test)
           
           
            temp = np.matmul(X_train.transpose(),X_train)
            
            n = temp.shape
            
            identity = np.identity(34)
            
            const = ()
            const = [alpha]
            mse_train_multiple = list()
            mse_test_multiple = list()
            for each in const:
                temp = temp + (each*identity)
                former = np.array(np.linalg.inv(temp))
                latter = np.matmul(X_train.transpose(),y_train)
                # print(latter.shape)
                theta = np.matmul(former,latter)
                
                print(" Coefficient matrix for alpha " +str(each) +"-"+str(theta))

                y_predict = np.matmul(X_train, theta)
                # print("Predicted Response for a multiple response features:", y_predict)
                y_predict_test = np.matmul(X_test, theta)

                mse_train_multiple.append(metrics.mean_squared_error(y_train, y_predict))
                mse_test_multiple.append(metrics.mean_squared_error(Y_test, y_predict_test))
            print ("\n#############TRAINING RESULTS##############")
            print("Mean Squared ERROR on TRAIN SET:{}".format(mse_train_multiple))
            print ("\n#############TESTING RESULTS##############")
            print("Mean Squared ERROR on TEST SET:{}".format(mse_test_multiple))
            return theta, mse_train_multiple, mse_test_multiple
        
        else:
            print("Method for Modelling with Regression not entered")
