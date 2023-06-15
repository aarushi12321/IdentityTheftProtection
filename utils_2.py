from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def create_sliding_window(data, sequence_length=10, stride=1):
    X_list, y_list = [], []
    for i in range(len(data)):
      if (i + sequence_length) < len(data):
        X_list.append(data.iloc[i:i+sequence_length:stride, :].values)
        y_list.append(data.iloc[i+sequence_length, -1])

    return np.array(X_list), np.array(y_list)

def get_train_test(data):

    train_split = 0.7
    n_train = int(train_split * len(data))
    n_test = len(data) - n_train

    features = ['day_of_week', 'hour_of_day', 'Energy_consumption']
    feature_array = data[features].values

    # Fit Scaler only on Training features
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(feature_array[:n_train])
    # Fit Scaler only on Training target values
    target_scaler = MinMaxScaler()
    target_scaler.fit(feature_array[:n_train, -1].reshape(-1, 1))

    # Transfom on both Training and Test data
    scaled_array = pd.DataFrame(feature_scaler.transform(feature_array),
                                columns=features)

    sequence_length = 10
    X, y = create_sliding_window(scaled_array, 
                                sequence_length)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, y_train, X_test, y_test, scaled_array, target_scaler

class BayesianLSTM(nn.Module):

    def __init__(self, n_features, output_length, batch_size):

        super(BayesianLSTM, self).__init__()

        self.batch_size = batch_size # user-defined

        self.hidden_size_1 = 128 # number of encoder cells (from paper)
        self.hidden_size_2 = 32 # number of decoder cells (from paper)
        self.stacked_layers = 2 # number of (stacked) LSTM layers for each stage
        self.dropout_probability = 0.5 # arbitrary value (the paper suggests that performance is generally stable across all ranges)

        self.lstm1 = nn.LSTM(n_features, 
                             self.hidden_size_1, 
                             num_layers=self.stacked_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=self.stacked_layers,
                             batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size_2, output_length)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        hidden = self.init_hidden1(batch_size)
        output, _ = self.lstm1(x, hidden)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        output = output[:, -1, :] # take the last decoder cell's outputs
        y_pred = self.fc(output)
        return y_pred
        
    def init_hidden1(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1))
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1))
        return hidden_state, cell_state
    
    def init_hidden2(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2))
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2))
        return hidden_state, cell_state
    
    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)

    def predict(self, X):
        return self(torch.tensor(X, dtype=torch.float32)).view(-1).detach().numpy()

def create_model(X_train, y_train, X_test, y_test, scaled_array):
    n_features = scaled_array.shape[-1]
    sequence_length = 10
    output_length = 1
    batch_size = 128
    learning_rate = 0.01

    bayesian_lstm = BayesianLSTM(n_features=n_features,
                             output_length=output_length,
                             batch_size = batch_size)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(bayesian_lstm.parameters(), lr=learning_rate)

    return bayesian_lstm, criterion, optimizer

def train_model(X_train, y_train,bayesian_lstm, criterion, optimizer):
    bayesian_lstm.train()
    n_epochs = 150
    batch_size = 128
    train_loss = []

    for e in range(1, n_epochs+1):
        for b in range(0, len(X_train), batch_size):
            features = X_train[b:b+batch_size,:,:]
            target = y_train[b:b+batch_size]    

            X_batch = torch.tensor(features,dtype=torch.float32)    
            y_batch = torch.tensor(target,dtype=torch.float32)

            output = bayesian_lstm(X_batch)
            loss = criterion(output.view(-1), y_batch)  

            loss.backward()
            optimizer.step()        
            optimizer.zero_grad() 

        if e % 10 == 0:
            train_loss.append(loss.item())
            print('epoch', e, 'loss: ', loss.item())

    return bayesian_lstm, train_loss

def inverse_transform(y,target_scaler):
  return target_scaler.inverse_transform(y.reshape(-1, 1))

def eval(data, bayesian_lstm, X_train, X_test,target_scaler):
    offset = 10
    train_split = 0.7
    n_train = int(train_split * len(data))

    training_df = pd.DataFrame()
    training_df['date'] = data['date'].iloc[offset:n_train + offset:1]
    training_predictions = bayesian_lstm.predict(X_train)
    training_df['Energy_consumption'] = inverse_transform(training_predictions,target_scaler)
    training_df['source'] = 'Training Prediction'

    training_truth_df = pd.DataFrame()
    training_truth_df['date'] = training_df['date']
    training_truth_df['Energy_consumption'] = data['Energy_consumption'].iloc[offset:n_train + offset:1] 
    training_truth_df['source'] = 'True Values'

    testing_df = pd.DataFrame()
    testing_df['date'] = data['date'].iloc[n_train + offset::1] 
    testing_predictions = bayesian_lstm.predict(X_test)
    testing_df['Energy_consumption'] = inverse_transform(testing_predictions, target_scaler)
    testing_df['source'] = 'Test Prediction'

    testing_truth_df = pd.DataFrame()
    testing_truth_df['date'] = testing_df['date']
    testing_truth_df['Energy_consumption'] = data['Energy_consumption'].iloc[n_train + offset::1] 
    testing_truth_df['source'] = 'True Values'

    evaluation = pd.concat([training_df, 
                        testing_df,
                        training_truth_df,
                        testing_truth_df
                        ], axis=0)
    
    return evaluation, testing_df, testing_truth_df

def get_test_uncertainty_df(testing_df, bayesian_lstm, X_test, target_scaler):
    n_experiments = 100

    test_uncertainty_df = pd.DataFrame()
    test_uncertainty_df['date'] = testing_df['date']

    for i in range(n_experiments):
        experiment_predictions = bayesian_lstm.predict(X_test)
        test_uncertainty_df['Energy_consumption_{}'.format(i)] = inverse_transform(experiment_predictions, target_scaler)

    energy_consumption_df = test_uncertainty_df.filter(like='Energy_consumption', axis=1)
    test_uncertainty_df['Energy_consumption_mean'] = energy_consumption_df.mean(axis=1)
    test_uncertainty_df['Energy_consumption_std'] = energy_consumption_df.std(axis=1)

    test_uncertainty_df = test_uncertainty_df[['date', 'Energy_consumption_mean', 'Energy_consumption_std']]

    test_uncertainty_df['lower_bound'] = test_uncertainty_df['Energy_consumption_mean'] - 3*test_uncertainty_df['Energy_consumption_std']
    test_uncertainty_df['upper_bound'] = test_uncertainty_df['Energy_consumption_mean'] + 3*test_uncertainty_df['Energy_consumption_std']


    return test_uncertainty_df

def get_return(test_uncertainty_plot_df, truth_uncertainty_plot_df):
    bounds_df = pd.DataFrame()

    # Using 99% confidence bounds
    bounds_df['lower_bound'] = test_uncertainty_plot_df['lower_bound']
    bounds_df['prediction'] = test_uncertainty_plot_df['Energy_consumption_mean']
    bounds_df['real_value'] = truth_uncertainty_plot_df['Energy_consumption']
    bounds_df['upper_bound'] = test_uncertainty_plot_df['upper_bound']

    bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
                            (bounds_df['real_value'] <= bounds_df['upper_bound']))
    
    return_statement = "Proportion of points contained within 99% confidence interval: "+str(np.round(bounds_df['contained'].mean(),2))

    return return_statement











    