import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

def pulldata(ticker):
	tickerdata = yf.Ticker(ticker)
	
	today = datetime.datetime.today().isoformat()
	 
	tickerDF = tickerdata.history(period = '1d', start='2016-1-1', end = today [:10])#pulling historic data from 2016 to today
	
	priceLast = tickerDF['Close'].iloc[-1]
	return tickerDF

df_total = pulldata('TSLA')

df = pd.DataFrame(df_total)

df1 = df.reset_index()['Close']

#Using minmax scalar to scale to values between 0 and 1 
scaler = MinMaxScaler(feature_range = (0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

#Split data for training and testing sets
split = 0.8 #80/20 split for training/test
n = len(df1)
size_train = int(n*split)
size_test = int(n-size_train)
train_data = df1[0:size_train,:]
test_data = df1[size_train:n,:1]

def create_data(df, window = 1):
	
	n = len(df)
	x_data = []
	y_data = []
	
	for i in range(n - (window + 1)):
		a = df[i : i+window, 0]
		x_data.append(a)
		y_data.append(df[window + i, 0])

	return np.array(x_data), np.array(y_data)

step_size = 100

x_test, y_test = create_data(test_data, step_size)
x_train, y_train = create_data(train_data, step_size)

print(x_train.shape)

x_train =x_train.reshape(x_train.shape[0],x_test.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)


#Building the LSTM

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences=True))
model.add(Dropout(0.2))   
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

filepath = 'savedmodels/stockpredictor.hdf5'

checkpoint = ModelCheckpoint(filepath = filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
history = model.fit(x_train,y_train, epochs = 100, validation_data = (x_test,y_test), callbacks = [checkpoint], verbose = 1, shuffle = False)

from keras.models import load_model

model = load_model(filepath)

train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

train_predict = scaler.inverse_transform(train_predict)
#y_train = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
#y_test = scaler.inverse_transform(y_test)

trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[step_size:len(train_predict)+step_size, :] = train_predict

testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(step_size*2)+1:len(df1)-1, :] = test_predict

df1 = scaler.inverse_transform(df1)

plt.plot(df1, 'g')
plt.plot(trainPredictPlot, 'r')
plt.plot(testPredictPlot,'b')
plt.show()
