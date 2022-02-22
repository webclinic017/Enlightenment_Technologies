from utilities import add_time_column, timefrom main_support import run_modelimport pandas as pddef main():	             # Print Start Time    start_time = time('start')         # Read in Dataframes    training_data = pd.read_csv('../data/TRAINING_DATA.txt').drop_duplicates()    testing_data = pd.read_csv('../data/TESTING_DATA.txt').drop_duplicates()        # Add Time Columns    add_time_column(training_data)    add_time_column(testing_data)        # Sort by Time    training_data = training_data.sort_values(by = ['time.index'], ascending = True)    testing_data = testing_data.sort_values(by = ['time.index'], ascending = True)        # Important Parameters    epoch_num = 10    batch_num = 25       training_set_len = .8    prediction_window = 2    predict_col_num = 0    num_of_predictions = 1    prediction_parameters = [epoch_num, batch_num, training_set_len, prediction_window, predict_col_num, num_of_predictions]        # Run the Models Specified    run_model(prediction_parameters, training_data, testing_data)           # Print End and Run Time    end_time = time('end')    time(end_time - start_time)    if __name__ == '__main__':	main()        