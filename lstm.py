import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop

# Function to fetch data from Yahoo Finance
def fetch_forex_data(currency_pair):
    symbol = f"{currency_pair[:3]}{currency_pair[3:]}=X"
    df = yf.download(symbol, start='2004-01-01', interval='1d')

    if not df.empty:
        df.reset_index(inplace=True)
        df.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close'
        }, inplace=True)

        df.drop(['Adj Close', 'Volume'], axis=1, inplace=True, errors='ignore')
        return df
    else:
        st.error("Error fetching data. Please try again later.")
        return None

# Function for 1-day prediction preprocessing
def preprocess_data_for_1_day(data, sequence_length=1, feature='Close'):
    data.dropna(inplace=True)

    # Convert Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Create a lag column for sequence prediction
    data['Lag_Close'] = data[feature].shift(sequence_length)

    # Drop rows with missing values after shifting
    data.dropna(inplace=True)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Lag_Close', feature]])

    # Prepare sequences (samples, timesteps, features)
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])  # Lagged values
        y.append(scaled_data[i, 1])  # Current price for next day

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM (samples, timesteps, features)

    # Train-test split (80-20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = data['Date'].values[train_size + sequence_length:]

    return X_train, X_test, y_train, y_test, test_dates, scaler

# Function for 1-week prediction preprocessing
def preprocess_data_for_1_week(data, sequence_length=1, feature='Close'):
    data.dropna(inplace=True)

    # Convert Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Create a lag column for sequence prediction
    data['Lag_Close'] = data[feature].shift(sequence_length)

    # Shift the target column by 7 days for 1-week prediction
    data['Target_Close'] = data[feature].shift(-7)

    # Drop rows with missing values after shifting
    data.dropna(inplace=True)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Lag_Close', feature]])

    # Prepare sequences (samples, timesteps, features)
    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - 7):  # Adjust for 7-day shift
        X.append(scaled_data[i - sequence_length:i, 0])  # Lagged values
        y.append(scaled_data[i + 7, 1])  # Price 7 days ahead

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM (samples, timesteps, features)

    # Train-test split (80-20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = data['Date'].values[train_size + sequence_length + 7:]

    return X_train, X_test, y_train, y_test, test_dates, scaler

# Function to build and train LSTM model with hyperparameters
def build_and_train_lstm_model(X_train, y_train, lstm_units=50, lstm_layers=1, dropout_rate=0.2, learning_rate=0.001, epochs=20, batch_size=32, optimizer='adam', loss_function='mean_squared_error', gradient_clip=None):
    model = Sequential()

    # Adding LSTM layers based on user input
    for i in range(lstm_layers):
        if i == lstm_layers - 1:
            model.add(LSTM(units=lstm_units, return_sequences=False, input_shape=(X_train.shape[1], 1)))
        else:
            model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1], 1)))

        model.add(Dropout(dropout_rate))  # Add dropout after each LSTM layer

    # Output layer
    model.add(Dense(1))

    # Choose optimizer
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate, clipvalue=gradient_clip) if gradient_clip else Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate, clipvalue=gradient_clip) if gradient_clip else RMSprop(learning_rate=learning_rate)

    # Compile model
    model.compile(optimizer=opt, loss=loss_function)

    # Train model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    return model

# Main function for Streamlit app
def main():
    st.title('Forex Price Prediction Using LSTM with Hyperparameter Tuning')

    # Step 1: Select currency pair
    currency_pair = st.selectbox('Select Currency Pair', ['GBPUSD', 'GBPEUR', 'EURUSD'])

    # Step 2: Fetch data from Yahoo Finance for the selected currency pair
    st.write(f"Fetching historical data for {currency_pair} starting from 2004...")
    data = fetch_forex_data(currency_pair)

    if data is not None:
        st.write(f"Data fetched successfully for {currency_pair}!")
        st.write(data.head())

        # Step 3: Hyperparameter selections
        prediction_type = st.selectbox('Select Prediction Type', ['Next Day', 'Next Week'])
        lstm_units = st.slider('Number of LSTM Units', min_value=10, max_value=200, step=10, value=50)
        lstm_layers = st.slider('Number of LSTM Layers', min_value=1, max_value=3, value=1)
        dropout_rate = st.slider('Dropout Rate', min_value=0.0, max_value=0.5, step=0.05, value=0.2)
        learning_rate = st.slider('Learning Rate', min_value=0.0001, max_value=0.01, step=0.0001, value=0.001)
        batch_size = st.slider('Batch Size', min_value=16, max_value=128, step=16, value=32)
        epochs = st.slider('Epochs', min_value=10, max_value=100, step=10, value=20)
        sequence_length = st.slider('Sequence Length (Time Steps)', min_value=1, max_value=30, step=1, value=1)
        optimizer = st.selectbox('Optimizer', ['adam', 'rmsprop'])
        loss_function = st.selectbox('Loss Function', ['mean_squared_error', 'mean_absolute_error'])
        gradient_clip = st.slider('Gradient Clipping Value', min_value=0.0, max_value=5.0, step=0.1, value=0.0)
        feature = st.selectbox('Select Input Feature', ['Close', 'High', 'Low'])

        # Preprocess the fetched data for LSTM
        if prediction_type == 'Next Day':
            X_train, X_test, y_train, y_test, test_dates, scaler = preprocess_data_for_1_day(data, sequence_length, feature)
        elif prediction_type == 'Next Week':
            X_train, X_test, y_train, y_test, test_dates, scaler = preprocess_data_for_1_week(data, sequence_length, feature)

        # Build and train the LSTM model with hyperparameters
        model = build_and_train_lstm_model(X_train, y_train, lstm_units=lstm_units, lstm_layers=lstm_layers, dropout_rate=dropout_rate, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, optimizer=optimizer, loss_function=loss_function, gradient_clip=gradient_clip)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(np.concatenate([X_test[:, 0, :], y_pred], axis=1))[:, 1]
        y_test = scaler.inverse_transform(np.concatenate([X_test[:, 0, :], y_test.reshape(-1, 1)], axis=1))[:, 1]

        # Calculate RMSE and MAPE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)

        st.write(f'{currency_pair} - {prediction_type} Prediction - RMSE: {rmse:.4f}, MAPE: {mape:.4f}')

        # Plot the results
        plt.figure(figsize=(14, 7))
        plt.plot(test_dates, y_test, color='blue', label=f'Actual {prediction_type} Prices ({currency_pair})')
        plt.plot(test_dates, y_pred, color='red', linestyle='dashed', label=f'Predicted {prediction_type} Prices ({currency_pair})')
        plt.title(f'{prediction_type} Price Prediction for {currency_pair}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)
    else:
        st.info("Please select a currency pair and ensure a valid API key.")

if __name__ == '__main__':
    main()
