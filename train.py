import pandas as pd
import pickle
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import numpy as np
# Load the dataset
with open("./data/ref_log_return.pkl", "rb") as f:
    train_log_return = pickle.load(f)


data = []
num_samples, num_hours, num_cryptos = train_log_return.shape

for i in range(num_samples):
    for crypto in range(num_cryptos):
        for hour in range(num_hours):
            data.append({
                'item_id': f'sample_{i}_crypto_{crypto}',  # Unique ID for each crypto in each sample
                'timestamp': pd.Timestamp(f'2024-01-01 {hour}:00:00'),  # Placeholder timestamp for each hour
                'log_return': train_log_return[i, hour, crypto]  # The log return value for that crypto at that hour
            })

# Convert to DataFrame
train_df = pd.DataFrame(data)

# Convert to TimeSeriesDataFrame format required by AutoGluon
train_data = TimeSeriesDataFrame.from_data_frame(
    df=train_df,
    id_column='item_id',
    timestamp_column='timestamp',


)

# Check the format of the processed DataFrame
print(train_data.head())

# Initialize the TimeSeriesPredictor
predictor = TimeSeriesPredictor(
    target='log_return',
    prediction_length=10,  # We want to predict the next 24 hours
    freq='H',
    path="model_dict",# Hourly data

)

# Train the model
predictor.fit(train_data,presets="medium_quality",)

predictions = predictor.predict(train_data)

# Initialize an empty array to hold the predictions in the original format
mean_predictions = predictions['mean']

# Reshape the mean predictions back into the original shape (8937, 24, 3)
num_samples = 8937
num_hours = 24
num_cryptos = 3

# Initialize an empty array for reshaped predictions
# reshaped_predictions = np.zeros((num_samples, num_hours, num_cryptos))
#
# # Loop through each sample and cryptocurrency to reshape the predictions correctly
# for i in range(num_samples):
#     for crypto in range(num_cryptos):
#         item_id = f'sample_{i}_crypto_{crypto}'
#
#         # Extract the mean predictions for this particular item_id
#         # You should now get only the 24-hour predictions for this item
#         pred = mean_predictions.loc[item_id].values[:24]  # Extract the 24 hours of predictions
#
#         # Place the 24-hour predictions into the reshaped array
#         reshaped_predictions[i, :, crypto] = pred
#
# # Check the shape to ensure it's correct
# print(reshaped_predictions.shape)  # Should be (8937, 24, 3)

# Save the reshaped predictions as 'fake_log_return.pkl'
with open("fake_log_return.pkl", "wb") as f:
    pickle.dump(predictions, f)
