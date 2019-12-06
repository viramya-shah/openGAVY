import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.tree import DecisionTreeRegressor

warnings.simplefilter("ignore")

class preprocess():

    def __init__(self):
        # random.seed(29)
        print ("PRE-PROCESSING METHODS LOADED..\n")

    def gavy_crosstab(self, data):
        for i in data.columns:
            print (pd.crosstab(data.loc[:,'TARGET'], data.loc[:, i], margins = True, normalize = True))    

    def gavy_string_to_categry(self, data):
        print("*** String to Category ***")
        df_all = data.copy()
        for col_name in df_all.columns:
            if df_all[col_name].dtypes == np.float64:
                df_all[col_name] = df_all[col_name].fillna(df_all[col_name].mean())
    
            if df_all[col_name].dtypes != np.float64 and df_all[col_name].dtypes != np.int64:
                df_all[col_name] = df_all[col_name].astype('category').cat.codes
                df_all[col_name] = df_all[col_name].fillna('0')
        return df_all


    def gavy_impute_missing_values(self, data, method = 'mean', constant = -1, feature_list = 0):
        print ("***\tFill missing values with mean/median/mode/constant: impute_missing_values ***")
        df_imputed_data = pd.DataFrame.copy(data)       
        to_impute_feature_list = []
        
        for col_name in df_imputed_data.columns:
            if df_imputed_data[col_name].isnull().sum()/len(df_imputed_data) > 0:
                to_impute_feature_list.append(col_name)
        
        for col_name in to_impute_feature_list:
            if method == 'mean':
                df_imputed_data[col_name] = df_imputed_data[col_name].fillna(df_imputed_data[col_name].mean())
            elif method == 'median':
                df_imputed_data[col_name] = df_imputed_data[col_name].fillna(df_imputed_data[col_name].median())
            elif method == 'mode':
                df_imputed_data[col_name] = df_imputed_data[col_name].fillna(df_imputed_data[col_name].mode())
            elif method == 'constant':
                df_imputed_data[col_name] = df_imputed_data[col_name].fillna(constant)
        
        if feature_list == 0:
            return df_imputed_data
        else:
            return df_imputed_data, to_impute_feature_list

    def gavy_binning_data(self, data, response_variable, threshol_unique =20, tuning = 1, max_depth = 10, min_sample_leaf = 200):
        print("*** Starting Binning Data.. ***")
        df = pd.DataFrame.copy(data)
        if df.isnull().any().any():
            print ("Cannot work without Null values. Please remove null values and then try again\n")
            return None

        col_names = df.columns
        col_name_continuos = []
        col_name_categorical = []

        for c_name in col_names:
            if df[c_name].nunique() > threshol_unique:
                col_name_continuos.append(c_name)
            elif df[c_name].nunique() <= threshol_unique:
                col_name_categorical.append(c_name)
        
        col_name_left = list(set(col_names) - set(col_name_categorical) - set(col_name_continuos))
        df_col_to_bin = df[col_name_continuos]
        df_col_not_to_bin = df[col_name_categorical]
        df_col_left = df[col_name_left]

        mat_col_to_bin = df_col_to_bin.as_matrix()
        mat_col_to_bin_col_names = df_col_to_bin.columns

        df_X = pd.DataFrame()
        dictionary_threshold = {}

        # for i in range(len(mat_col_to_bin)):
        # i = -1
        for i in range(len(mat_col_to_bin_col_names)):
            # i += 1
            x = mat_col_to_bin[:, i]
            number_of_unique = len(np.unique(x))
            # print ("Binning variable:", str(i+1), "/", str(len(mat_col_to_bin_col_names)),"\t", mat_col_to_bin_col_names[i])

            '''
            Add grid search function.
            will work currently for tuning = 0
            '''

            if tuning == 0:
                clf_dt_regressor = DecisionTreeRegressor(
                    max_depth = max_depth, 
                    min_samples_leaf = min_sample_leaf,
                    max_leaf_nodes = 10,
                    random_state = 29
                )

                # print (x.shape)

                clf_dt_regressor.fit(np.swapaxes(np.array([x]),0,1).reshape(-1,1), x.reshape(-1,1))
                '''
                missing y value => target variable
                i guess CF assumes that the target is at last..and thus .shape[:-1]
                '''
                pred = clf_dt_regressor.predict(x.reshape(-1,1))

                threshold_out = np.sort(np.unique(clf_dt_regressor.tree_.threshold[clf_dt_regressor.tree_.feature > -2]))

                if len(threshold_out) == 0:
                    dictionary_threshold.update({mat_col_to_bin_col_names[i] : np.unique(pred)})
                else:
                    dictionary_threshold.update({mat_col_to_bin_col_names[i]: threshold_out})
                
                x1 = pd.DataFrame(pred)
                df_X = pd.concat([df_X, x1], axis = 1)
                df_X = df_X.rename(columns = {
                    df_X.columns[i] : mat_col_to_bin_col_names[i]
                })

        for c in col_name_categorical:
            label_encoder = LabelEncoder()
            df_col_not_to_bin[c] = label_encoder.fit_transform(df_col_not_to_bin[c])
            
            mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
            dictionary_threshold.update({c : np.array([*mapping])})
        
        data_binned = pd.concat([df_col_not_to_bin, df_X, df_col_left], axis = 1)
        print("\tBinning Completed")
        return data_binned, col_name_continuos, dictionary_threshold

    def gavy_information_values(self, data, target, iv_lower_bound = 0.02, iv_higher_bound = 10):
        print("***Starting information value computation***")
        X = pd.DataFrame(data).copy()
        y = X.loc[:, target]

        res_woe = []
        res_iv = []
        
        var = 1 # this is the positive class

        for i in range(X.shape[1]):
            
            x = X.iloc[:, i]
            woe_dict, iv = self.gavy_woe_single(x, y, var)
            res_woe.append(woe_dict)
            res_iv.append(iv)

        df_num_of_bins = pd.DataFrame(X.apply(pd.Series.nunique))
        df_num_of_bins.columns=['No of Bins']

        # col_name_wo_target = X.loc[:, X.columns != target].columns
        col_name_wo_target = X.columns

        iv_values = pd.DataFrame(np.stack((col_name_wo_target, res_iv)).T)
        iv_values[1] = pd.to_numeric(iv_values[1])
        iv_values = iv_values.set_index(iv_values[0])
        iv_values.drop([0], axis = 1, inplace = True)
        iv_values.columns= ['IV']

        df_iv_bins = pd.concat([df_num_of_bins, iv_values], axis = 1)   #All IVs and No of bins
        df_iv_bins = pd.DataFrame(df_iv_bins.sort_values(by='IV', ascending= False))
        df_iv_bins['variable_response'] = df_iv_bins.index
        df_iv_bins['Use_Case'] = df_iv_bins['IV'].apply(lambda x: "Considered" if x > iv_lower_bound and x < iv_higher_bound else "Rejected")
        df_iv_bins = df_iv_bins.rename(columns={'variable_response': 'variable'})
        
        iv_cols = df_iv_bins.loc[df_iv_bins['Use_Case'] == 'Considered']
        iv_cols = iv_cols['variable'].tolist()
        iv_cols.append('TARGET')

        print("***Computing IV completed***")
        return df_iv_bins, iv_cols


    def gavy_woe_single(self, x, y, var):

        total_event = (y == var).sum()
        total_non_event = (y != var).sum()

        x_unique = np.unique(x)
        woe_dict, iv = {}, 0

        x = x.to_frame()
        x.columns = ['temp']        

        for x_ in x_unique:
            # print (x.index[x['TARGET'] == x_)
            y_ = y.iloc[x.index[x['temp'] == x_].tolist()]

            count_event = (y_ == var).sum()
            count_non_event = (y_ != var).sum()

            rate_event = 1.0*count_event/total_event
            rate_non_event = 1.0*count_non_event/total_non_event

            if rate_event == 0.0:
                woe_ = -20.0
            elif rate_non_event == 0.0:
                woe_ = 20.0
            else:
                woe_ = math.log(rate_event/rate_non_event)
            
            woe_dict[x_] = woe_
            iv += (rate_event-rate_non_event)*woe_
        
        return woe_dict, iv