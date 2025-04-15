import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the stock ticker (GOOGL)
ticker = 'GOOGL'

# Load historical data for the last 1278 days
data = yf.download(ticker, period='1278d', interval='1d')
data.columns = data.columns.droplevel(1)
data.reset_index(inplace=True)

dataset_train = data.iloc[:1258]
training_set = data.iloc[:1258, 4:5].values

# Characteristic scaling (normalization)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))  # Scaling the values between 0 and 1
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):  # Using 60 previous steps to predict the following value
    X_train.append(training_set_scaled[i-60:i, 0])  # Input data for each instant
    y_train.append(training_set_scaled[i, 0])  # Output labels (next value)
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the entrances to the shape required by the RNN.
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Add feature dimension

# Part 2 - Construction of the RNN

# Import Keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize the RNN
regressor = Sequential()

# Adding the first LSTM layer and regularizing with Dropout
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))  # LSTM Layer
regressor.add(Dropout(0.2))  # Dropout to avoid overfiting

# Adding a second LSTM layer and Dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and Dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and Dropout
regressor.add(LSTM(units = 50)) 
regressor.add(Dropout(0.2))

# Adding an output layer
regressor.add(Dense(units = 1))  # Dense layer with a single output unit

# Compilando la RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')  # Using the Adam optimizer and mean squared error

# Fit the RNN to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)  # Training the network with 100 epochs and batch size of 32


# Part 3 - Making predictions and visualizing the results

# Obtaining the actual price of GOOGL stock
dataset_test = data.iloc[1258:]
real_stock_price = dataset_test.iloc[:, 4:5].values  # Actual price of stocks

# Obtaining the predicted price of GOOGL stock
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)  # Combining training and test data
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values  # Selecting the last 60 values

inputs = inputs.reshape(-1,1)  # Reshaping the input data
inputs = sc.transform(inputs)  # Scaling the input data, because train values are scaled

X_test = []
for i in range(60, 80):  # We use 60 previous values to make the prediction
# The maximum range is 80 because we forecast for 20 days.
    X_test.append(inputs[i-60:i, 0])  # Preparing the input data
X_test = np.array(X_test)  # Convert to array
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Reshaping for the RNN

predicted_stock_price = regressor.predict(X_test)  # Making predictions
predicted_stock_price = sc.inverse_transform(predicted_stock_price)  # De-escalating predictions

# Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Google Stocks Actual Price')  # Plotting actual prices
plt.plot(predicted_stock_price, color = 'blue', label = 'Google Stocks Predicted Price')  # Graphing the predictions
plt.title('Google Stocks Price Predictions')  # Title
plt.xlabel('Time')  # X
plt.ylabel('Google Stocks Price')  # Y
plt.legend()  # Legend
plt.show()
