from sklearn.model_selection import train_test_split
import numpy as np
from numpy import save, load
import pandas as pd
import json as json

df = pd.read_csv('Preprocessed_data.gzip', compression='gzip')

def get_data_and_labels(data):
    """
  This function provides separated data and labels

  input:
  data - entire unseparated dataset (pd.df)
  output:
  data, labels (pd.df, pd.df)
  """

    labels = data.iloc[:, 1:2]
    data = data.iloc[:, 2:]

    return data, labels


# divide the orignal df into data, labels
data, labels = get_data_and_labels(df)

# split Train dataset and Test dataset with ratio
x_train_wide, x_test_wide, y_train, y_test = train_test_split(data.values, labels.FLAG.values,
                                                              test_size=0.2, random_state = 2017)

x_sample = np.round(x_test_wide[18],4)
y_sample = y_test[18]

#print(y_sample)
np.save('/Users/aarushisethi/Desktop/PredOnly/samples/x_sample_5.npy', x_sample)

# c = np.fromfile('test2.dat', dtype=int)
# dtype -> 'float64', 'int64'