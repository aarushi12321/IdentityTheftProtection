from sklearn.model_selection import train_test_split
import numpy as np
from numpy import save, load
import pandas as pd

df = pd.read_csv('Preprocessed_data.gzip', compression='gzip')

def get_data_and_labels(data):

    labels = data.iloc[:, 1:2]
    user_id_df = pd.DataFrame(data=data.iloc[:, :1])
    data = pd.concat([user_id_df,data.iloc[:, 2:]], axis=1)
    data.rename(columns = {'CONS_NO':'userid'}, inplace = True)

    return data, labels


# divide the orignal df into data, labels
data, labels = get_data_and_labels(df)

# split Train dataset and Test dataset with ratio
x_train_wide, x_test_wide, y_train, y_test = train_test_split(data.values, labels.FLAG.values,
                                                              test_size=0.2, random_state = 2017)

y_test = np.reshape(y_train, (y_train.shape[0],1))
col_names_x = ['userid'] + list(np.arange(0,1034,1))
col_names_y = list(labels.columns)
x_df = pd.DataFrame(x_train_wide, columns = col_names_x)
y_df = pd.DataFrame(y_train, columns = col_names_y)
y_df = pd.concat([x_df.iloc[:,:1], y_df], axis=1)

x_df.to_csv('x_train.csv', index=False, header=False)
y_df.to_csv('y_train.csv', index=False, header=False)
