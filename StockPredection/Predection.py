from datetime import date

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
import streamlit as st
import yfinance as yf

#step1:dataset
#Fetch historical stock data from yahoo Finance
symbol = 'BTC'
start_date = '2020-01-01'
end_date = date.today()

dataset_train = yf.download(symbol, start=start_date, end=end_date)
dataset_train.to_csv('stock_data.csv')
training_set = dataset_train.iloc[:, 1:2].values
print(dataset_train.head())
#Data normalization
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []

for i in range(60, dataset_train.shape[0]):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



#Step2:model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error')
#step3:train
model.fit(X_train, y_train, epochs=10, batch_size=32)

#step4:test
# url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv'
dataset_test = pd.read_csv('stock_data.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
#Data test transformation
dataset_total = pd.concat((dataset_test['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []

for i in range(60, 60+dataset_test.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Data prediction
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#data test visualisation
# plt.plot(real_stock_price, color='black', label ='Stock Price')
# plt.plot(predicted_stock_price, color='green', label='Predicted Stock Price')
# plt.title(symbol + ' Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel(symbol+' Stock Price')
# plt.legend()
# plt.show()

# Streamlit App
st.title(symbol + ' Stock Price Prediction')
# st.line_chart(pd.DataFrame({
#     'Real Stock Price': real_stock_price.flatten(),
#     'Predicted Stock Price': predicted_stock_price.flatten()
# }))
chart_data = pd.DataFrame({
    'Date': dataset_test['Date'],
    'Real Stock Price': real_stock_price.flatten(),
    'Predicted Stock Price': predicted_stock_price.flatten()
}).set_index('Date')

st.line_chart(chart_data)


#save the model
model.save(symbol+'Model.keras')