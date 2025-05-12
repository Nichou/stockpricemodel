import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

import os
import logging
import json

class StockPriceModel:

    def __init__(self):
    
        self.x_train = np.array([])
        self.y_train = np.array([])
        self.df = pd.DataFrame()
        self.config = {}
    
    def load_configurations(self):
        
        config_path = 'config.json'
        
        if not os.path.exists('config.json'):
            logging.error("Non existent \'config.json\' file")
            return
        
        with open(config_path) as f:
            self.config = json.load(f)
    
    def load_data(self, p):
        
        if not os.path.exists(p):
            logging.error("Non existent path")
            return
        
        self.df = pd.read_csv(p)
        
    def preprocess(self, input_layer):
        
        data = self.df.filter(['Close'])
        if data.size == 0:
            logging.error("Non existent \'close\' colunm")
            return
        
        dataset = data.values
        training_data_len = math.ceil(len(dataset)*.8)
        
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        
        train_data = scaled_data[0:training_data_len, :]
        raw_x_train = []
        raw_y_train = []
        
        for i in range(input_layer, len(train_data)):
            raw_x_train.append(train_data[i-input_layer:i, 0])
            raw_y_train.append(train_data[i, 0])
            
        self.x_train = np.array(raw_x_train);
        self.y_train = np.array(raw_y_train); 
        
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))
        
        return training_data_len, dataset, scaler, scaled_data

    def build_model(self):
        
        model = Sequential()
    
        model.add(LSTM(50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
        
    def predict_and_compare(self):
    
        self.load_configurations()
        input_layer = self.config['input_layer']
    
        training_data_len, dataset, scaler, scaled_data = self.preprocess(input_layer)
    
        model = self.build_model()
        model.fit(self.x_train, self.y_train, batch_size=1, epochs=5)
        
        original = scaler.inverse_transform(scaled_data)
        result = []
        
        y_test = dataset[training_data_len:, :]
        
        last_prediction = 0;
        last_y = 0;
        
        up_correlation = 0;
        ups = 0;
        
        profit_handler = 0.0;
        
        for e in range(0, input_layer): #
        
            test_data = scaled_data[training_data_len-(input_layer+1)+e:training_data_len+e, :] #   
            x_test = []

            for i in range(input_layer, len(test_data)):
                x_test.append(test_data[i-input_layer:i, 0])
                
            x_test = np.array(x_test)
            
            predictions = model.predict(x_test) 
            predictions = scaler.inverse_transform(predictions)
            
            result.append(predictions[0][0])
            
            if y_test[e][0] > last_y:
                ups+=1;
            
                if predictions[0][0] > last_prediction:
                    up_correlation+=1;
            
            if predictions[0][0] > last_y and e > 0:
                profit_handler+=(y_test[e][0]/last_y)-1.0;            
            
            last_prediction = predictions[0][0]
            last_y = y_test[e][0]
        
        
        data = self.df.filter(['Close'])
        train = data[:training_data_len]
        valid = data[training_data_len:training_data_len+input_layer]
        valid['Predictions'] = result
        
        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Data', fontsize=18)
        plt.ylabel('Close price USD', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Values', 'Predicitions'], loc='lower right')
        plt.show()
        
        print("up correlation: "+str(100.0*(up_correlation/ups))+"%")
        print("profits handled: "+str(100.0*profit_handler)+"%")
        
       
        
        