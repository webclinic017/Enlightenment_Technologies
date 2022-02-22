from keras.layers import Dropout, Dense, LSTM, Conv1D, Flatten
from keras_self_attention import SeqSelfAttention
from tensorflow import keras

import numpy as np

class CNN_SelfAtten_LSTM_Model(keras.Model): 
    '''
    This model is adapted from: https://arxiv.org/pdf/2102.09024.pdf
    '''
    def __init__(self, input_size):
        super(CNN_SelfAtten_LSTM_Model, self).__init__()

        
        self.Conv1D_in = Conv1D(filters=128, kernel_size=3, strides=1, padding="causal", activation="relu", input_shape=input_size)
        
        self.Conv1D_1 = Conv1D(filters=128, kernel_size=3, strides=1, padding="causal", activation="relu")
        self.Conv1D_2 = Conv1D(filters=128, kernel_size=3, strides=1, padding="causal", activation="relu")
        self.Conv1D_3 = Conv1D(filters=128, kernel_size=3, strides=1, padding="causal", activation="relu")
        
        #Adding the LSTM layers and some Dropout regularisation
        self.LSTM_1 = LSTM(units = 128, return_sequences = True)
        self.Dropout_1 = Dropout(0.2)   
        
        self.LSTM_2 = LSTM(units = 128, return_sequences = True)
        self.Dropout_2 = Dropout(0.2)   
        
        # Adding a SeqSelfAttention
        self.SeqSelfAttention_1 = SeqSelfAttention(attention_activation='sigmoid')
        
        # Adding the GlobalAveragePooling1D
        self.Flatten = Flatten()
        
        self.Dense_1 = Dense(128, activation="relu")
        self.Dense_2 = Dense(64, activation="relu")
        self.Dense_3 = Dense(32, activation="relu")
        self.Dense_4 = Dense(1, activation="relu")
        
        
    def call(self, inputs):  
        
        Conv1D_in = self.Conv1D_in(inputs)
        
        Conv1D_1 = self.Conv1D_1(Conv1D_in)
        Conv1D_2 = self.Conv1D_2(Conv1D_1)
        Conv1D_3 = self.Conv1D_3(Conv1D_2)
        
        LSTM_1 = self.LSTM_1(Conv1D_3)
        Dropout_1 = self.Dropout_1(LSTM_1)
        
        LSTM_2 = self.LSTM_2(Dropout_1)
        Dropout_2 = self.Dropout_2(LSTM_2)
        
        SeqSelfAttention_1 = self.SeqSelfAttention_1(Dropout_2)
        
        Flatten = self.Flatten(SeqSelfAttention_1)
        
        Dense_1 = self.Dense_1(Flatten)
        Dense_2 = self.Dense_2(Dense_1)
        Dense_3 = self.Dense_3(Dense_2)
        Dense_4 = self.Dense_4(Dense_3)
        
        return Dense_4 
    
def train_model(model, input_shape, validation, X_train, y_train, epoch_num, batch_num, sc, num_data, i, prediction_num):
    
    model.build((None, X_train.shape[1], num_data))
    
    if i == 0:
        
        model.summary()
        
    print('\nPrediction Run {} of {}\n'.format(i + 1, prediction_num))
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    history = model.fit(X_train, y_train, epochs = epoch_num, validation_data = validation, batch_size = batch_num)
    
    return history
    

def stock_forcast(model, X_test, close_scale, selfpredict, prediction_window):
    
    if selfpredict == False:
    
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = close_scale.inverse_transform(predicted_stock_price)
    
    
    else:
        
        predicted_length = X_test[:,0].shape[0]
        predicted_stock_price = np.zeros((predicted_length,1))
        
        temp = X_test[0,:]
        
        early_end = predicted_length
        
        # Predict Future Values Using Past Predicted Values
        for i in range(0, early_end):#predicted_length):
            
            if i % 100 == 0:
                
                print('{} Remaining Predictions'.format(predicted_length - i))  
            
            temp = np.reshape(temp, (1, prediction_window, 1))
            prediction = model.predict(temp)[0]
            predicted_stock_price[i] = prediction
            temp = np.reshape(temp, (prediction_window, 1))
            shift_temp = temp
            
            for j in range(0, prediction_window):
                
                if j < prediction_window - 1:
                    
                    temp[j] = shift_temp[j + 1]
                    
                if j == prediction_window - 1:     
                    
                    temp[j] = prediction
            
        predicted_stock_price = close_scale.inverse_transform(predicted_stock_price)        
    
    return predicted_stock_price







