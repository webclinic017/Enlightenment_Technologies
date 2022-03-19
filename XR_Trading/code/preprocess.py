from sklearn.preprocessing import MinMaxScaler

import numpy as np

import math

def data_preprocessing(training_set_len, prediction_window, training_data):
        
    all_data = training_data
    
    # Get Index to be Date        
    all_data.set_index('time.index', inplace=True, drop=True, append=False)
    
    print(all_data.head())
    
    for col in all_data.columns:
        print(col)
    
    # Create a Dataframe With Only the SO2 Column
    s02_values = all_data.filter(['so2'])
    
    # Convert the dataframe to a numpy array to train the LSTM model
    target = all_data.values
    so2_target = s02_values.values
    training_data_len = math.ceil(len(target)* training_set_len)
    
    # Get Number of Data
    num_data = target.shape[1] - 1
    
    # Scaling
    sc = MinMaxScaler(feature_range=(0,1))
    so2_scale = MinMaxScaler(feature_range=(0,1))
    
    training_scaled_data = sc.fit_transform(target)
    so2_training_scaled_data = so2_scale.fit_transform(so2_target)

    train_data = training_scaled_data[0:training_data_len, : ]

    # Getting the predicted stock price
    test_data = training_scaled_data[training_data_len - prediction_window: , : ]

    return all_data, target, train_data, test_data, training_data_len, sc, so2_scale, num_data


def get_train_data(prediction_window, train_data, predict_col_num, num_of_data):

    train_len = len(train_data)
    train_vec_len = train_len - prediction_window
    num_of_data = train_data.shape[1]
    
    X_train = np.zeros((train_vec_len, prediction_window, num_of_data - 1))
    y_train = np.zeros((train_vec_len, 1))
    
    for i in range(0, train_vec_len):
        
        y_train[i] = train_data[i + prediction_window, predict_col_num]
        
        for j in range(0, prediction_window):
            
            for k in range(0, num_of_data - 1):
                
                X_train[i, j, k] = train_data[i + j, k + 1]
        
    return X_train, y_train 

def get_test_data(prediction_window, training_data_len, test_data, target, predict_col_num, num_of_data):
    
    test_len = len(test_data)
    test_vec_len = test_len - prediction_window
    num_of_data = test_data.shape[1]
    
    X_test = np.zeros((test_vec_len, prediction_window, num_of_data - 1))
    y_test =  np.zeros((test_vec_len, 1))
    
    for i in range(0, test_vec_len):
        
        y_test[i] = test_data[i + prediction_window, predict_col_num]
        
        for j in range(0, prediction_window):
            
            for k in range(0, num_of_data - 1):
                
                X_test[i, j, k] = test_data[i + j, k + 1]
        
    return X_test, y_test 






