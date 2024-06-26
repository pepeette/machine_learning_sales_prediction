from taipy import Gui
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_excel('supermarkt_sales.xlsx',engine="openpyxl",
    sheet_name="Sales",skiprows=3)

# add weekday data
df["Weekday"] = (pd.to_datetime(df["Date"], format="%D:%M:%Y").dt.dayofweek).apply(lambda x: calendar.day_name[x])

def generate_predictions(df):
    # Extract features (X) and target variable (y)
    grouped_data = df.groupby(["Weekday", "City", "Customer_type"]).agg({"Total": "sum"}).reset_index()

    # Extract features (X) and target variable (y)
    X = grouped_data[['Weekday', 'City', 'Customer_type']].copy()
    y = grouped_data['Total'].values

    # Create a column transformer to handle categorical features
    categorical_features = ['City', 'Customer_type', 'Weekday']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', OneHotEncoder(), categorical_features),
        ],
        remainder='passthrough'
    )

    # Create a pipeline with the preprocessor and model
    model = RandomForestRegressor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    predictions = []

    # Iterate over each unique combination of "Weekday," "City," and "Customer_type"
    for _, row in X.iterrows():
        # Filter the data for the current combination
        filter_condition = (df["Weekday"] == row["Weekday"]) & (df["City"] == row["City"]) & (df["Customer_type"] == row["Customer_type"])
        filtered_data = df[filter_condition]

        # Train the model for the current combination
        X_train = filtered_data[['Weekday', 'City', 'Customer_type']]
        y_train = filtered_data['Total'].values

        pipeline.fit(X_train, y_train)

        # Make predictions for the last row of the current combination
        new_data = row.to_frame().T
        prediction = pipeline.predict(new_data)
        predictions.append(prediction[0])

    # Create a new DataFrame with the predictions
    predictions_df = pd.DataFrame({"Predicted_Total": predictions})

    # Merge the original DataFrame with the predictions DataFrame
    df_with_predictions = pd.concat([df, predictions_df], axis=1)

    return df_with_predictions

# initialization of variables
cities = list(df["City"].unique())
types = list(df["Customer_type"].unique())


city = cities
customer_type = types

show_pred ="OFF"

page = """
<center><h2> **Sales DASHBOARD** </h2></center>

<|layout|columns = 20 80|gap=30px|
<sidebar|
**Filters** available:<br/>

. city<br/>
<|{city}|selector|lov={cities}|multiple|label=Select the City|dropdown|on_change=on_filter|width=100%|>

. customer<br/>
<|{customer_type}|selector|lov={types}|multiple|label=Select the Customer Type|dropdown|on_change=on_filter|width=100%|>

. Prediction<br/>
<|{show_pred}|toggle|lov=OFF;ON|>

|sidebar>

<|main|

<|Sales Table|expandable|expanded=False|
<|{df_selection}|table|width=100%|page_size=5|rebuild|class_name=table|>
|>

<|{sales_by_weekday}|chart|x=Weekday|y=Total|type=bar|title=Sales by Day of Week|render={show_pred=='OFF'}|>

<|{predictions_by_weekday}|chart|x=Weekday|y[1]=Total|y[2]=Total|type[1]=bar|type[2]=line|mode=markers|title=Sales by Day of Week|render={show_pred=='ON'}|>

<br/>

<|layout|columns=1 1|
<|sales tot|
#### Total sales:
#### US $ <|{int(df_selection["Total"].sum())}|>
|>
<|sales avg|
#### Average Sales Per Transaction:
#### US $ <|{round(df_selection["Total"].mean(), 2)}|>
|>
|>
|>

|>

"""


def filter(city, customer_type):
    df_selection = df[
        df["City"].isin(city)
        & df["Customer_type"].isin(customer_type)
    ]
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_selection["Weekday"] = pd.Categorical(df_selection["Weekday"], categories=weekday_order, ordered=True)
    sales_by_weekday = df_selection.groupby("Weekday").agg({"Total": "sum"})
    df_selection_with_predictions = generate_predictions(df_selection)
    predictions_by_weekday = df_selection_with_predictions.groupby("Weekday").agg({"Total": "sum"})
    
    # # Call the function to generate predictions (replace this with your actual prediction logic)
    # df_selection_with_predictions = generate_predictions(df_selection)
    
    # # Calculate predicted sales by weekday
    # predictions_by_weekday = df_selection_with_predictions.groupby("Weekday")["Predicted_Total"].sum()
    
    # # Add the predictions to the sales_by_weekday DataFrame
    # sales_by_weekday["Predicted_Total"] = predictions_by_weekday
    
    sales_by_weekday["Weekday"] = sales_by_weekday.index
    predictions_by_weekday["Weekday"] = predictions_by_weekday.index
    return df_selection, sales_by_weekday, predictions_by_weekday


def on_filter(state):
    state.df_selection, state.sales_by_weekday, state.predictions_by_weekday= filter(
        state.city, state.customer_type)
    #state.pred = generate_predictions(state.sales_by_weekday)

if __name__ == "__main__":
    # initialize dataframes
    df_selection,sales_by_weekday,  predictions_by_weekday= filter(city, customer_type)
    Gui(page).run(use_reloader=True)