import pandas as pd
from sklearn.preprocessing import scale
import tensorflow
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.utils import to_categorical
import numpy as np
class stock_predict(object):
    def __init__(self):
        self.dir = 'stock_data/'
        stock_code = '7203_2019'#toyota motor stock data on 2019
        term = 5
        
        
        #load data and preprocessing
        data = self.data_load(stock_code)
        master = self.data_preprocessing(data)
        print(master)
        
        print('unique target number')
        print(master['return_target'].value_counts())
        
        _x,_y = master[:-1],master.pop('return_target').shift(-1)[:-1].values

        #split data (train test)
        test_days = 30
        x_train_data = _x[:-test_days]
        y_train_data = _y[:-test_days]
        x_test_data = _x[-test_days:]
        y_test_data = _y[-test_days:]
        
        x_train_data.to_csv('x_train_data.csv')
        x_test_data.to_csv('x_test_data.csv')
        #convert to 5term sequencial data
        seq_x_train = self.data_to_sequence(x_train_data,term)
        seq_x_test  = self.data_to_sequence(x_test_data,term)
        seq_y_train  = to_categorical(np.array(y_train_data))[term-1:]
        seq_y_test  = to_categorical(np.array(y_test_data))[term-1:]

        #make model
        model = self.make_models(seq_x_train.shape[1:])
        #train model
        history = model.fit(seq_x_train,seq_y_train,epochs=50,
                            batch_size=20,validation_split=0.2)
        self.plotter(history)

    def plotter(self,history):
        import matplotlib.pyplot as plt
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def make_models(self,input_size):
        print(input_size)
        hidden_units = 16
        model = Sequential()
        model.add(LSTM(hidden_units,dropout=0.1,recurrent_dropout=0.5,
                       input_shape=input_size,return_sequences=True,
                       kernel_regularizer=regularizers.l2(0.001)))
        model.add(LSTM(hidden_units,dropout=0.1,recurrent_dropout=0.5,
                       kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dense(32,activation="relu",kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dense(2,activation="sigmoid",kernel_regularizer=regularizers.l2(0.001)))
        model.compile(optimizer="rmsprop",
                      loss="categorical_crossentropy",
                      metrics=['acc'])
        return model
    def data_load(self,stock_num):
        dir = self.dir+str(stock_num)+'.csv'
        return pd.read_csv(dir,encoding='shift_jis',index_col='日付',skiprows=[0])

    def data_preprocessing(self,data,normalize=True):
        data['出来高'] = np.log(data['出来高'].replace(0,1)/1000 + 1)
        data['return'] = data['終値'].diff()
        #sma processing
        data['sma_5'] = data['終値'].rolling(5).mean()
        data['sma_15'] = data['終値'].rolling(15).mean()
        data['sma_25'] = data['終値'].rolling(25).mean()
        
        #normalize data
        if normalize:
            r = data['return'].copy()
            data = pd.DataFrame(scale(data),columns=data.columns,
                                index=data.index)
            data['return_target'] = 0
            data['return_target'].loc[r>0] = 1
        #remove nan data
        data = data.replace((np.inf,-np.inf),np.nan).dropna()
        return data
    def data_to_sequence(self,data,term):
        feat_data = np.zeros((len(data.index) - term + 1,term,data.shape[-1]))
        for i in range(len(data.index)-term+1):
            feat_data[i] = data[i:i+term]
        return feat_data

if __name__ == "__main__":
    pred = stock_predict()
