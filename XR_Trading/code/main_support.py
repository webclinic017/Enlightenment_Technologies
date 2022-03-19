from rnn_model import CNN_SelfAtten_LSTM_Model, train_model, forcast, future_forcast
from preprocess import data_preprocessing, get_train_data, get_test_data
from plotting import visualize_fit, visualize_loss, visualize_val_loss
from utilities import print_shape

import numpy as np
    
def run_model(prediction_parameters, training_data, testing_data):

    plot_other_predictions = True

    # Important Parameters 
    epoch_num = prediction_parameters[0]
    batch_num = prediction_parameters[1]
    training_set_len = prediction_parameters[2]
    prediction_window = prediction_parameters[3]
    predict_col_num = prediction_parameters[4]
    prediction_num = prediction_parameters[5]
    
    # Process Data
    all_data, target, train_data, test_data, training_data_len, sc, so2_scale, num_data = data_preprocessing(training_set_len, prediction_window, training_data)

    # Get Test and Train Data
    X_train, y_train = get_train_data(prediction_window, train_data, predict_col_num, num_data)
    X_test, y_test = get_test_data(prediction_window, training_data_len, test_data, target, predict_col_num, num_data)
    
    # Print the Shape of the Arrays
    print_shape(all_data, train_data, test_data, X_train, y_train, X_test, y_test)

    # Train Model Parameters
    input_shape = (X_train.shape[1], num_data)
    validation = (X_test, y_test)
    
    # Vectors for Each Prediction
    prediction_len = X_test.shape[0]
    multiple_predictions = np.zeros((prediction_len, prediction_num))
    multiple_loss = np.zeros((epoch_num, prediction_num))
    multiple_val_loss = np.zeros((epoch_num, prediction_num))
    
    #print(input_shape)
    
    # Begin Training and Instantiating Model
    model = CNN_SelfAtten_LSTM_Model(input_shape)
    #model = CNN_LSTM_Model(input_shape)
    #model = LSTM_Model(input_shape)
    
    # Run for the Number of Total Predictions
    for i in range(0, prediction_num): 
    
        history = train_model(model, input_shape, validation, X_train, y_train, epoch_num, batch_num, sc, num_data, i, prediction_num)

        print('X_test Shape', X_test.shape)

        #Making predictions using the test dataset
        predicted_so2 = forcast(model, X_test, so2_scale)

        # Run the function to illustrate accuracy and loss
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        for j in range(0, prediction_len):
            
            multiple_predictions[j,i] = predicted_so2[j]
            
        for k in range(0,epoch_num):
            
            multiple_loss[k, i] = loss[k]
            multiple_val_loss[k, i] = val_loss[k]
            
    # Visualize Results
    visualize_val_loss(multiple_val_loss, prediction_num)
    visualize_loss(multiple_loss, prediction_num)
    visualize_fit(all_data, training_data_len, multiple_predictions, prediction_num, plot_other_predictions)
    
    # Make Predictions for Future
    future_forcast(model, testing_data, so2_scale, prediction_window, sc)    
    
    
    
    
    
    
    
 
