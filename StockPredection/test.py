from datetime import date
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# Function to load and preprocess data
def load_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Failed to download data for {symbol}: {e}")
        return None

# Function to create and train the LSTM model
def create_and_train_model(X_train, y_train):
    st.info("Training the model. Please wait...")
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
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    st.success("Model training complete!")

    return model

# Function to make predictions on the test set
def make_predictions(model, X_test, scaler):
    st.info("Making predictions on the test set. Please wait...")
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    st.success("Predictions complete!")
    return predicted_stock_price

# Streamlit App
st.set_page_config(
    page_title="Stock Price Prediction App",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Sidebar with user inputs
st.sidebar.title("Stock Prediction Settings")
symbol = st.sidebar.text_input("Enter Stock Ticker Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

# Load and preprocess data
data = load_data(symbol, start_date, end_date)

# Main content
st.title("Stock Price Prediction App")
st.header("Predicting future stock prices")

# Check if data is available
if data is not None:
    # Display dataset information
    st.subheader('Dataset Information:')
    st.dataframe(data.head())

    # Data normalization
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prepare training data
    X_train, y_train = [], []

    for i in range(60, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Create and train the model
    model = create_and_train_model(X_train, y_train)

    # Data test transformation
    dataset_total = pd.concat((data['Close'], data['Close']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(data) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []

    for i in range(60, 60 + len(data)):
        X_test.append(inputs[i - 60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Make predictions
    predicted_stock_price = make_predictions(model, X_test, sc)

    # Plot actual and predicted stock prices
    st.subheader(f'{symbol} Stock Price Prediction')
    chart_data = pd.DataFrame({
        'Actual Stock Price': data['Close'].values[-len(predicted_stock_price):],
        'Predicted Stock Price': predicted_stock_price.flatten()
    })
    st.line_chart(chart_data)

    # Display additional charts or information
    st.header("Additional Information")
    # Add more charts or insights as needed

else:
    st.error("No data available. Please check the ticker symbol or date range.")
