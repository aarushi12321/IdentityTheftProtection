from tqdm import tqdm
import pickle
import numpy as np
from numpy import save, load
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score

from tensorflow import keras
from keras.utils import np_utils
from keras.layers import Input, concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model    
from keras.optimizers import SGD

df = pd.read_csv('Preprocessed_data.gzip', compression='gzip')

df.head()


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

def turn_compatible(x_train_wide, x_test_wide, y_train, y_test):
  """
  This function turns the input data compatible with the settings
  input:
  x_train_wide - x_train dataset values (array)
  x_test_wide - x_test dataset values (array)
  Output:
  x_train_deep, x_test_deep (array, array)
  """
  # turn it to compatible data

  x_train_wide = x_train_wide[:,:7*int(x_train_wide.shape[1]/7)]
  x_test_wide = x_test_wide[:,:7*int(x_test_wide.shape[1]/7)]

  print(f'X train shape: {x_train_wide.shape}')
  print(f'Y train shape: {y_train.shape}')
  print(f'X test shape: {x_test_wide.shape}')
  print(f'Y test shape: {y_test.shape}')
  print('')

  return x_train_wide, x_test_wide


def reshape_to_deep(x_train_wide, x_test_wide):
    """
    This function reshapes the wide data to deep data.

    Input:
    x_train_wide - x_train dataset values (array)
    x_test_wide - x_test dataset values (array)
    Output:
    x_train_deep, x_test_deep (array, array)
    """
    x_train_deep = x_train_wide.reshape(x_train_wide.shape[0], 1, -1, 7)
    x_test_deep = x_test_wide.reshape(x_test_wide.shape[0], 1, -1, 7)

    print("This is all users' data")
    print(x_train_deep.shape)

    print("This is one user's data")
    print(x_train_deep[0].shape)

    print("This is one user's data split by weeks")
    print(x_train_deep[0][0].shape)
    print(x_train_deep[0][0])

    print("This is one user's one week's data")
    print(x_train_deep[0][0][0].shape)
    print(x_train_deep[0][0][0])

    x_train_deep = x_train_deep.transpose(0, 2, 3, 1)
    x_test_deep = x_test_deep.transpose(0, 2, 3, 1)

    return x_train_deep, x_test_deep


def expand_data(data):
    d1 = np.zeros([data.shape[0] * 3, data.shape[1], data.shape[2]])

    for i in range(data.shape[0]):
        if i >= (data.shape[0] - 2):
            d1[i * 3:(i * 3 + data.shape[0] - i), :, :] = data[i:, :, :]
        else:
            d1[i * 3:(i * 3) + 3, :, :] = data[i:i + 3, :, :]

    d2 = np.zeros([d1.shape[0], d1.shape[1] * 3, d1.shape[2]])

    for j in range(d1.shape[1]):
        if j >= (d1.shape[1] - 2):
            d2[:, j * 3:(j * 3 + d1.shape[1] - j), :] = d1[:, j:, :]
        else:
            d2[:, j * 3:(j * 3) + 3, :] = d1[:, j:(j + 3), :]

    return d2

def preprocess_kernel(data):
  data1 = np.zeros(data.shape)
  data2 = np.zeros(data.shape)

  for i in range(int(data.shape[0]/3)):
    k = data[(i*3):(i*3+3),:,:]
    data1[i*3,:,:] = 2*k[0,:,:] - k[1,:,:] - k[2,:,:]
    data1[i*3+1,:,:] = 2*k[1,:,:] - k[0,:,:] - k[2,:,:]
    data1[i*3+2,:,:] = 2*k[2,:,:] - k[0,:,:] - k[1,:,:]

  for i in range(int(data.shape[1]/3)):
    k = data[:,(i*3):(i*3+3),:]
    data1[:,i*3,:] = 2*k[:,0,:] - k[:,1,:] - k[:,2,:]
    data1[:,i*3+1,:] = 2*k[:,1,:] - k[:,0,:] - k[:,2,:]
    data1[:,i*3+2,:] = 2*k[:,2,:] - k[:,0,:] - k[:,1,:]

  return data1 + data2


def self_define_cnn_kernel_process(data):
    """
    1. expand data from (x, y, z) to (x*3, y*3, z) (Because Conv2D convolution with stride (3,3) for our preprocess)

    2. 3*3 kernel process:

        [2*V_1 - V_2 - V3
          2*V_2 - V_1 - V3
          2*V_3 - V_1 - V2]
        +
        [2*Vt_1 - Vt_2 - Vt_3, 2*Vt_2 - Vt_1 - Vt_3, 2*Vt_3 - Vt_1 - Vt_2]

    input: data (array)
    output: final_data (array)
    """
    # initialize the final data matrics with all zeros
    data_final = np.zeros([data.shape[0], data.shape[1] * 3, data.shape[2] * 3, data.shape[3]])
    for i in tqdm(range(data.shape[0]), desc='Converting to CNN'):
        d1 = data[i, :, :, :]
        d1_expand = expand_data(d1)
        d1_final = preprocess_kernel(d1_expand)
        data_final[i, :, :, :] = d1_final
    print(data_final.shape)
    return data_final


def Wide_CNN(weeks, days, channel, wide_len, lr=0.005, decay=1e-5, momentum=0.9):
    inputs_deep = Input(shape=(weeks * 3, days * 3, channel))
    inputs_wide = Input(shape=(wide_len,))

    x_deep = Conv2D(32, (3, 3), strides=(3, 3), padding='same', kernel_initializer='he_normal')(inputs_deep)
    x_deep = MaxPooling2D(pool_size=(3, 3))(x_deep)
    x_deep = Flatten()(x_deep)
    x_deep = Dense(128, activation='relu')(x_deep)

    x_wide = Dense(128, activation='relu')(inputs_wide)

    x = concatenate([x_wide, x_deep])
    x = Dense(64, activation='relu')(x)

    pred = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[inputs_wide, inputs_deep], outputs=pred)

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics='accuracy')
    return model


class MyMetric(keras.callbacks.Callback):

    def __init__(self, num, train_ratio, validation_data):
        self.train_ratio = train_ratio
        self.num = num
        self.epoch = 0
        self.validation_data = validation_data

    def accuracy(self, y_true, y_pred):

        y_pred_label = np.argmax(y_pred, axis=1)
        bool_arr = y_pred_label==y_true
        sum = np.sum(bool_arr)
        acc = sum/(bool_arr.shape[0])

        return acc

    def precision_at_k(self, r, k):

        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    def average_precision(self, r):

        r = np.asarray(r) != 0
        out = [self.precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
        if not out:
            return 0.
        return np.mean(out)

    def mean_average_precision(self, rs):

        return np.mean([self.average_precision(r) for r in rs])

    def on_epoch_end(self, batch, logs={}):
        self.epoch += 1
        preds = self.model.predict(self.validation_data[0][0:2])
        y = self.validation_data[1]

        acc = self.accuracy(y,preds)
        auc = roc_auc_score(y, preds)

        temp = pd.DataFrame(
            {'label_0': list(y), 'label_1': list(1 - y), 'preds_0': list(preds), 'preds_1': list(1 - preds)})

        map1 = self.mean_average_precision([list(temp.sort_values(by='preds_0', ascending=0).label_0[:100]),
                                            list(temp.sort_values(by='preds_1', ascending=0).label_1[:100])])
        map2 = self.mean_average_precision([list(temp.sort_values(by='preds_0', ascending=0).label_0[:200]),
                                            list(temp.sort_values(by='preds_1', ascending=0).label_1[:200])])

        print('Acc: %.4f    AUC:%.4f     MAP@100:%.4f      MAP@200:%.4f  \n' % (acc, auc, map1, map2))

        log = 'Epoch:%2d    Acc: %.4f   AUC:%.4f     MAP@100:%.4f      MAP@200:%.4f  \n' % (
        self.epoch, acc, auc, map1, map2)

print('Reading data and label...')
print('')

# divide the orignal df into data, labels
data, labels = get_data_and_labels(df)

# split Train dataset and Test dataset with ratio
x_train_wide, x_test_wide, y_train, y_test = train_test_split(data.values, labels.FLAG.values,
                                                              test_size=0.2, random_state = 2017)

print(f'X train shape: {x_train_wide.shape}')
print(f'Y train shape: {y_train.shape}')
print(f'X test shape: {x_test_wide.shape}')
print(f'Y test shape: {y_test.shape}')
print('')

print(f'x_train_wide looks like: {x_train_wide[0,:].shape}')
print(x_train_wide[0,:])
print('')

print('Changing x_train_wide to a compatible shape')
x_train_wide, x_test_wide = turn_compatible(x_train_wide, x_test_wide, y_train, y_test)

print('Reshaping the data for a deep network...')
print('')

x_train_deep, x_test_deep = reshape_to_deep(x_train_wide, x_test_wide)
weeks, days, channel = x_train_deep.shape[1], x_train_deep.shape[2], 1
wide_len = x_train_wide.shape[1]

print('Defining data for cnn network...')
print('')
x_train_pre = self_define_cnn_kernel_process(x_train_deep)
x_test_pre = self_define_cnn_kernel_process(x_test_deep)

print('Starting training...')
print('')

for i in tqdm(range(1), desc='Rounds'):
    model = Wide_CNN(weeks, days, channel, wide_len)

    if i == 0:
        print(model.summary())

    model.fit([x_train_wide, x_train_pre], y_train, batch_size=64, epochs=30, verbose=1,
              validation_data=([x_test_wide, x_test_pre], y_test))
              # callbacks=[MyMetric(i, 0.8, ([x_test_wide, x_test_pre], y_test))]

pickle.dump(model, open('model1.pkl','wb'))

