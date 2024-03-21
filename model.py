import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import load_model
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_excel('supermarkt_sales.xlsx',engine="openpyxl",
    sheet_name="Sales",skiprows=3)

def split_data(data, group_col='Date', test_size=0.2, random_state=42):
    # Drop unnecessary column if it exists
    df = data.drop(columns=['Unnamed: 0'], errors='ignore')

    # Group by specified column
    grouped_data = df.groupby(group_col)

    # Initialize empty lists for training and testing data
    train_data_list = []
    test_data_list = []

    # Iterate over groups
    for _, group in grouped_data:
        # Split the group into train and test (80-20 split)
        train_group, test_group = train_test_split(group, test_size=test_size, random_state=random_state)

        # Append the split groups to the respective lists
        train_data_list.append(train_group)
        test_data_list.append(test_group)

    # Create DataFrames from the lists
    train_data = pd.concat(train_data_list)
    test_data = pd.concat(test_data_list)

    return train_data, test_data

train_data, test_data = split_data(data)

# Data Preprocessing
train_data['Date'] = pd.to_datetime(train_data['Date'])
test_data['Date'] = pd.to_datetime(test_data['Date'])

train_data.set_index('Date', inplace=True)
test_data.set_index('Date', inplace=True)

train_data['day'] = train_data.index.day
train_data['month'] = train_data.index.month
train_data['year'] = train_data.index.year
train_data['day_of_week'] = train_data.index.dayofweek

test_data['day'] = test_data.index.day
test_data['month'] = test_data.index.month
test_data['year'] = test_data.index.year
test_data['day_of_week'] = test_data.index.dayofweek

# Model Preparation and Training
common_cols = ['Unit price','Quantity','Total', 'day', 'month', 'year', 'day_of_week']
scaler = MinMaxScaler()
scaler.fit(train_data[common_cols])

scaled_train_data = scaler.transform(train_data[common_cols])
scaled_test_data = scaler.transform(test_data[common_cols])

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 10
X_train, y_train = prepare_data(scaled_train_data, n_steps)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))

# Function to create model
def create_model(optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(LSTM(50, activation=activation, input_shape=(n_steps, X_train.shape[2])))
    model.add(Dense(X_train.shape[2]))  # Output layer with the same number of features
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Define the model
model = KerasRegressor(build_fn=create_model, verbose=0)

# Define the grid search parameters
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [32, 64],  # Explore different batch sizes
    'epochs': [50, 100],  # Vary the number of epochs
}
# Create Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2)
grid_result = grid.fit(X_train, y_train)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps, X_train.shape[2]), return_sequences=True))
model.add(LSTM(50, activation='relu'))
model.add(Dense(X_train.shape[2]))
model.compile(optimizer='rmsprop', loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)

# Test Data Preparation and Prediction
def prepare_test_data(data, n_steps):
    X = []
    for i in range(len(data) - n_steps + 1):
        X.append(data[i:i + n_steps])
    return np.array(X)

X_test = prepare_test_data(scaled_test_data, n_steps)

# Ensure that the length of X_test matches the original length of scaled_test_data
diff_len = len(scaled_test_data) - len(X_test)
X_test = np.pad(X_test, ((0, diff_len), (0, 0), (0, 0)), 'constant', constant_values=np.nan)

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

predicted_sales = model.predict(X_test)
predicted_sales = scaler.inverse_transform(predicted_sales)

print(predicted_sales)

# Add the predicted sales to the test dataset
test_data['predicted_sales'] = predicted_sales
# Save the predictions to a CSV file with 'id' and 'predicted_sales' columns
test_data.reset_index(inplace=True)  # Reset index to access 'date' as a column

predicted_df = test_data[['Date', 'Total', 'predicted_sales']]  # Adjust column names if needed

# Save the predictions to a CSV file with 'id' and 'predicted_sales' columns
predicted_df.to_csv('predicted_sales.csv', index=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['predicted_sales'], label='Predicted Sales', marker='x')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Predicted Sales')
plt.legend()
plt.show()

model.save('sales_prediction_model.h5')