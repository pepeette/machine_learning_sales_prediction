from importlib import invalidate_caches
invalidate_caches()
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

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess(data):
    data['day'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['year'] = data['Date'].dt.year
    data['day_of_week'] = data['Date'].dt.dayofweek

    data = data[["City", "Customer_type", "Product line", "Quantity","Date", "day_of_week","Total"]]

    # # Encode categorical variables
    # le_city = LabelEncoder()
    # data['City'] = le_city.fit_transform(data['City'])

    # le_customer_type = LabelEncoder()
    # data['Customer_type'] = le_customer_type.fit_transform(data['Customer_type'])

    # le_product_line = LabelEncoder()
    # data['Product line'] = le_product_line.fit_transform(data['Product line'])

    # le_day_of_week = LabelEncoder()
    # data['month'] = le_day_of_week.fit_transform(data['month'])

    # # Scale numerical variables
    # scaler_quantity = MinMaxScaler()
    # data['Quantity'] = scaler_quantity.fit_transform(data[['Quantity']])

    # scaler_total = MinMaxScaler()
    # data['Total'] = scaler_total.fit_transform(data[['Total']])

    # Drop the original 'Date' column
    #data = data.drop(columns=['Date'])

    return data


def split_data(data, group_col='Date', test_size=0.2, random_state=42):
    # Drop unnecessary column if it exists
    df = preprocess(data.copy())
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

    # Save train and test data to CSV files without indexes
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)

    return train_data, test_data


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def train_and_predict(train_data, test_data, city_col='City', day_of_week_col='day_of_week', target_col='Total'):
    # Initialize the linear regression model
    model = LinearRegression()

    # Initialize empty lists to store predictions and actual values
    predictions = []
    actual_values = []

    # Iterate over unique cities
    for city in train_data[city_col].unique():
        # Filter data for the current city
        train_city_data = train_data[train_data[city_col] == city]
        test_city_data = test_data[test_data[city_col] == city]

        # Extract features and target for training
        X_train = train_city_data[[day_of_week_col]]
        y_train = train_city_data[target_col]

        # Fit the model
        model.fit(X_train.values.reshape(-1, 1), y_train)

        # Extract features for testing
        X_test = test_city_data[[day_of_week_col]]

        # Make predictions
        y_pred = model.predict(X_test.values.reshape(-1, 1))

        # Store predictions and actual values
        predictions.extend(y_pred)
        actual_values.extend(test_city_data[target_col])

    # Calculate and print the root mean squared error (RMSE)
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    print(f'Root Mean Squared Error (RMSE): {rmse}')

    return predictions


#split
train_data, test_data = split_data(data)

# Train and predict using the created function
predictions = train_and_predict(train_data, test_data)

# Add predictions to the test_data DataFrame
test_data['Predicted_Total'] = predictions

# Save the test_data with predictions to a CSV file
test_data.to_csv('test_data_with_predictions.csv', index=False)

# import matplotlib.pyplot as plt

# # Group the test_data by day_of_week and City to plot the actual and predicted values
# grouped_test_data = test_data.groupby(['day_of_week', 'City']).agg({'Total': 'sum', 'Predicted_Total': 'sum'}).reset_index()

# # Plot the actual values
# plt.figure(figsize=(12, 6))
# for city in grouped_test_data['City'].unique():
#     city_data = grouped_test_data[grouped_test_data['City'] == city]
#     plt.plot(city_data['day_of_week'], city_data['Total'], label=f'Actual - {city}', marker='o')

# # Plot the predicted values
# for city in grouped_test_data['City'].unique():
#     city_data = grouped_test_data[grouped_test_data['City'] == city]
#     plt.plot(city_data['day_of_week'], city_data['Predicted_Total'], label=f'Predicted - {city}', linestyle='dashed', marker='o')

# plt.xlabel('Day of Week')
# plt.ylabel('Total Sales')
# plt.title('Actual vs Predicted Total Sales per Day of Week')
# plt.legend()
# plt.show()
