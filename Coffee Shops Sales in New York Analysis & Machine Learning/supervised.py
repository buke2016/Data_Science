import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor




original_df = pd.read_csv('data/PairIDdata.csv')

# Assuming 'original_df' is your original DataFrame with the 'transaction_time' column

# Convert 'transaction_time' to datetime format if it's not already in datetime format
original_df['transaction_date'] = pd.to_datetime(original_df['transaction_date'])

# Extract year, month, day, and day of the week from the 'transaction_time' column
original_df['date'] = original_df['transaction_date']
original_df['year'] = original_df['transaction_date'].dt.year
original_df['month'] = original_df['transaction_date'].dt.month
original_df['day_of_month'] = original_df['transaction_date'].dt.day
original_df['day_of_week'] = original_df['transaction_date'].dt.dayofweek

# Group by year, month, day of the month, and day of the week, and count the number of rows
grouped_df = original_df.groupby(['date', 'year', 'month', 'day_of_month', 'day_of_week']).size().reset_index(name='orders_per_day')

# Plot the number of rows per day
'''
plt.figure(figsize=(12, 6))
plt.plot(grouped_df['date'], grouped_df['orders_per_day'])
plt.xlabel('Date')
plt.ylabel('Orders per Day')
plt.title('Orders per Day over time')
plt.show()
'''

# Use only 3 features. Month, day of the month, and day of the week. Its all the same year so we don't need that. and we don't need the date since its encapsulated in the other 3 features
ml_ready_df = grouped_df[['month', 'day_of_month', 'day_of_week', 'orders_per_day']]
X = ml_ready_df.drop('orders_per_day', axis=1)
y = ml_ready_df['orders_per_day']


#X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)

# Initialize GBM regressor
gbm_regressor = GradientBoostingRegressor(n_estimators=75)

# Train GBM regressor
gbm_regressor.fit(X, y)

# Make predictions on training and validation sets
y_train_pred = gbm_regressor.predict(X)
#y_val_pred = gbm_regressor.predict(X_test)

# Calculate RMSE for training set
rmse_train = np.sqrt(mean_squared_error(y, y_train_pred))

# Calculate RMSE for validation set
#rmse_val = np.sqrt(mean_squared_error(y_test, y_val_pred))

print("Training RMSE:", rmse_train)
#print("Test RMSE:", rmse_val)

# Lets graph the predictions vs. the actual values
'''
plt.figure(figsize=(12, 6))
plt.plot(grouped_df['date'], y_train_pred, color='red', label='Predicted Orders per Day')
plt.plot(grouped_df['date'], grouped_df['orders_per_day'], color='blue', label='Actual Orders per Day')
plt.xlabel('Date')
plt.ylabel('Orders per Day')
plt.title('Predicted Orders per Day over time')
plt.legend()
plt.show()
'''

# what about future values? lets predict the next 30 days
# Month is 1-12
# Day of the month is 1-31
# Day of the week is 0-6, 0 is Monday, 6 is Sunday
customers = gbm_regressor.predict([[7, 1, 5]])
print("Predicted number of customers on July 1st, 2020:", customers)

# This little test above demonstrates how overfit my current model is. The model doesn't seem to consider the day of the week at all, which is an important feature. 

'''
estimators = [50, 60, 75, 100, 150, 200, 500, 1000]
for num_estimators in estimators:
    avg_rmse_train = 0
    avg_rmse_val = 0
    for i in range(100):
        #split the test_val set into test and validation (20% each)

        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3)

        # Initialize GBM regressor
        gbm_regressor = GradientBoostingRegressor(n_estimators=num_estimators)

        # Train GBM regressor
        gbm_regressor.fit(X_train, y_train)

        # Make predictions on training and validation sets
        y_train_pred = gbm_regressor.predict(X_train)
        y_val_pred = gbm_regressor.predict(X_val)

        # Calculate RMSE for training set
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

        # Calculate RMSE for validation set
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

        #print("Training RMSE:", rmse_train)
        #print("Validation RMSE:", rmse_val)
        avg_rmse_train += rmse_train
        avg_rmse_val += rmse_val
    avg_rmse_train /= 100
    avg_rmse_val /= 100
    print("Number of Estimators:", num_estimators)
    print("Average Training RMSE:", avg_rmse_train)
    print("Average Validation RMSE:", avg_rmse_val)
    print("\n")

    # Now I want to see how well the model generalizes to the test set
'''
