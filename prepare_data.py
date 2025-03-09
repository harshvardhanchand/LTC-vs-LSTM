import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load dataset (first 20,000 rows)
df = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv').iloc[:20000]

# Handle missing values properly
df['pm2.5'] = df['pm2.5'].ffill().bfill()

# Encode cyclical time features
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

# One-hot encode wind direction properly
wind_encoder = OneHotEncoder(sparse_output=False)
wind_encoded = wind_encoder.fit_transform(df[['cbwd']])
wind_df = pd.DataFrame(wind_encoded, columns=wind_encoder.get_feature_names_out(['cbwd']))
df = pd.concat([df, wind_df], axis=1)

# Create lag features
for lag in [1, 3, 6, 12, 24]:
    df[f'pm25_lag_{lag}'] = df['pm2.5'].shift(lag)

# Create rolling statistics
for window in [6, 12, 24, 48]:
    df[f'pm25_rolling_mean_{window}'] = df['pm2.5'].rolling(window=window).mean().shift(1)
    df[f'temp_rolling_mean_{window}'] = df['TEMP'].rolling(window=window).mean().shift(1)

# Drop rows with NaN (from lag features)
df.dropna(inplace=True)

# Select features (all columns you want to scale)
features = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'] + \
           [col for col in df.columns if 'wind_' in col] + \
           [col for col in df.columns if 'lag_' in col or 'rolling_' in col]

# Target is just pm2.5
target = ['pm2.5']

# Fit a scaler on features
features_scaler = MinMaxScaler()
data_scaled = features_scaler.fit_transform(df[features].values)

# Fit a separate scaler on target
target_scaler = MinMaxScaler()
target_scaled = target_scaler.fit_transform(df[target].values)

# Replace the target column in the scaled features with the separately scaled target
# (Assuming pm2.5 is the first column in features)
data_scaled[:, 0] = target_scaled.flatten()

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Target: pm2.5 (already scaled by target_scaler)
    return np.array(X), np.array(y)

seq_length = 24  # One-day window
X, y = create_sequences(data_scaled, seq_length)

# Train/Test Split (70-30)
train_size = int(len(X) * 0.7)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Save dataset along with both scalers' parameters
np.savez('dataset.npz',
         X_train=X_train, y_train=y_train,
         X_test=X_test, y_test=y_test,
         features_scaler_min=features_scaler.min_,
         features_scaler_scale=features_scaler.scale_,
         target_scaler_min=target_scaler.min_,
         target_scaler_scale=target_scaler.scale_)
