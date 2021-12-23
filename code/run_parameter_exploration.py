from main_support import run_model, run_multi_model
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np

def run(prediction_parameters, ticker, dates, specifications, models, which_run):
    # Important Parameters 
    batch_num_vector = prediction_parameters[1]
    prediction_window_vector = prediction_parameters[3]
    
    # Make Vectors to Eventually Make Plot With 
    batch_num_len = batch_num_vector[1] - batch_num_vector[0]
    prediction_window_len = prediction_window_vector[1] - prediction_window_vector[0]
    
    # Make Vectors for Plotting
    batch_num_axis = np.zeros(batch_num_len).reshape(batch_num_len,1)
    prediction_window_axis = np.zeros(prediction_window_len).reshape(prediction_window_len,1)
    parameter_exploration_array = np.zeros(batch_num_len*prediction_window_len).reshape(batch_num_len,prediction_window_len)
    
    single_run = which_run[0]
    multi_model = which_run[1]
    category_run = which_run[2]
    
    for i in range(batch_num_vector[0], batch_num_vector[1]):
        
        batch_num_axis[i - batch_num_vector[0]] = i
        prediction_parameters[1] = i
        
        for j in range(prediction_window_vector[0], prediction_window_vector[1]):
            
            prediction_window_axis[j - prediction_window_vector[0]] = j
            prediction_parameters[3] = j
    
            # Single Model Run
            if single_run == True:
                
                L2_Score = run_model(prediction_parameters, ticker, dates, specifications, models[2])
                
            # Run All Six Variations 
            if multi_model == True:    
                
                L2_Score = run_multi_model(prediction_parameters, ticker, dates, specifications)
            
            # Run the Model
            if category_run == True:
            
                for j in range(0, 2):
                
                    single_ticker = ticker[j]
                    
                    print('\n\n###################### Starting ' + single_ticker + ' Analysis ######################\n\n')
                    
                    for i in range(0,3):
                    
                        print('\n#########  Starting ' + models[i] + ' Run #########\n')
                        L2_Score += run_model(prediction_parameters, ticker, dates, specifications, models[i])
                
                L2_Score = L2_Score/len(ticker)
                        
            print(L2_Score)
            parameter_exploration_array[i - prediction_window_vector[0], j - prediction_window_vector[0]] = L2_Score
            
    ax = plt.axes(projection ='3d')
    ax.plot_wireframe(batch_num_axis, prediction_window_axis, parameter_exploration_array, color ='green')
    ax.set_title('wireframe geeks for geeks')