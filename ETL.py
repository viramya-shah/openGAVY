import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import metrics
import warnings
warnings.simplefilter("ignore")

class ETL():
    def __init__(self):
        print("ETL METHODS LOADED..")    

    def gavy_read_csv(self, loc, columns = '', delim = None):
        print("*** Read and write files of format ./csv and ./xls ***")
        if columns is None or columns is '':
            return pd.read_csv(loc, delimiter=delim)
        else:
            return pd.read_csv(loc, usecols = columns, delimiter=None)
    

    def gavy_write_csv(self, file_name, data, columns = '', delim=None):
        print("*** Read and write files of format ./csv and ./xls ***")
        if columns is None or columns is '':
            data.to_csv(file_name, delimiter = delim)
        else:
            data.to_csv(file_name, columns = columns, delimiter = delim)

    def gavy_read_excel(self,loc, columns = '', delim =  None):
        print("*** Read and write files of format ./csv and ./xls ***")
        if columns is None or columns is '':
            return pd.read_excel(loc, delimeter = delim)
        else:
            return pd.read_excel(loc, usecols = columns, )
    
    def gavy_write_excel(self, file_name, data, columns = [], delim = None):
        print("*** Read and write files of format ./csv and ./xls ***")
        if columns is None or columns is '':
            return data.to_excel(file_name, delimiter = delim)
        else:
            return data.to_excel(file_name, columns = columns, delimiter = delim)

    def gavy_partition(self, data, target_var, test_size = 0.25, random_seed = 1087):
        '''
        #################################################################################################
        ## Use this function to partition the dataset into train and test parts #########################
        #################################################################################################
        #Input:
            #data: the dataframe
            #target_var: The name of the target variable
            #test_size: Proportion of the test data 
        
        #Output:
            #df_train: training set
            #df_test: testing set

        #Example:
            #df_train, df_test = partition_dataset(data, target_var, test_size = 0.25)
            
        # using a stratified randomness
        ###################################################################################################
        '''
        print("*** Partition the data into test and train datasets ***")
        X = (data.loc[:,data.columns != target_var])
        X_names = np.array(X.columns)
        X = np.array(X)
        
        df_target = data[target_var]
        y = np.array(df_target)
        np.random.seed(random_seed)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state  = random_seed)
        
        X_train = pd.DataFrame(X_train)
        X_train.columns = X_names
        y_train = pd.DataFrame(y_train)
        y_train.columns = [target_var]
        
        X_test = pd.DataFrame(X_test)
        X_test.columns = X_names
        y_test = pd.DataFrame(y_test)
        y_test.columns = [target_var]
        
        # X_train = X_train.drop(data.columns[0], axis = 1)
        # X_test = X_test.drop(data.columns[0], axis = 1)

        df_train = pd.concat([X_train, y_train], axis = 1)
        df_test = pd.concat([X_test, y_test], axis = 1)
        
        return df_train, df_test

    def gavy_standarization(self, data, method = ''):
        # Standarization of the columns : can be done with mean, mode and median
        data_std = pd.DataFrame.copy(data)
        feature_list = list(data_std.columns.values)

        feature_list_NA = []

        print('Standardizing features...\n')
        print('Please wait while the system is Standarizing Features...')
        features_standardized = []
        for var in feature_list:
            str_filter = type(data_std[var][0])
            if ('int' in str(str_filter) or ('float' in str(str_filter))):  #Only if the feature is int or float then can the feature be standardized. Note that there can be multiple types of int and float
                features_standardized.append(var)
                if data_std[var].std() != 0:
                    if method == 'mean_std':
                        data_std[var] = (data_std[var] - data_std[var].mean())/data_std[var].std()
                    elif method == 'min_max':
                        data_std[var] = (data_std[var] - data_std[var].min())/(data_std[var].max() - data_std[var].min())
                    elif method == 'standard_scaler':
                        data_scale = preprocessing.StandardScaler().fit(data_std)
                        data_std = data_scale.transform(data_std)
                    else:
                        print("No standardization was done, please select a method: mean_std or min_max or standard_scaler")
        print("Feature Standarization completed")

        return data_std, features_standardized


    



