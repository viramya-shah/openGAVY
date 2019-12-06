import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import metrics

data = pd.read_csv("./unit_test.csv")
target_var = 'TARGET'
X = (data.loc[:,data.columns != target_var])
X_names = np.array(X.columns)
X = np.array(X)

df_target = data[target_var]
y = np.array(df_target)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

X_train = pd.DataFrame(X_train)
X_train.columns = X_names
y_train = pd.DataFrame(y_train)
y_train.columns = [target_var]

X_test = pd.DataFrame(X_test)
X_test.columns = X_names
y_test = pd.DataFrame(y_test)
y_test.columns = [target_var]


X_train = X_train.drop(data.columns[0], axis = 1)


X_test = X_test.drop(data.columns[0], axis = 1)


# print(X_train)
# print(y_train)


df_train = pd.concat([X_train, y_train], axis = 1)
print(df_train)
df_test = pd.concat([X_test, y_test], axis = 1)
print(df_test)

