import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === Load and preprocess data ===
df = pd.read_csv(r"C:\Users\zeyad\Documents\Well_Ventilated_CO2_Dataset.csv")  # Correct path to new dataset
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')  # Correct format for new dataset
df = df.sort_values(by='DateTime')

# === Feature engineering ===
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month
df['Weekday'] = df['DateTime'].dt.weekday
df['Is_Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

# We will focus on the 'CO2 (PPM)' for prediction but you can include other features.
features = ['Hour', 'Day', 'Month', 'Weekday', 'Is_Weekend', 'CO2 (PPM)', 'TVOC (PPB)']

threshold_high = df['CO2 (PPM)'].quantile(0.95)
threshold_low = df['CO2 (PPM)'].quantile(0.05)
df = df[(df['CO2 (PPM)'] >= threshold_low) & (df['CO2 (PPM)'] <= threshold_high)]

# === Normalize data ===
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# === Create sequences for LSTM ===
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :-1])  # All columns except the target column 'CO2 (PPM)'
        y.append(data[i+seq_len, -2])  # Target is the 'CO2 (PPM)' column
    return np.array(X), np.array(y)

sequence_length = 48  # 24 hours of 30-min intervals

# Ensure that we have enough data for the sequences
if len(df_scaled) <= sequence_length:
    raise ValueError("Not enough data for the given sequence length.")

X, y = create_sequences(df_scaled[features].values, sequence_length)

# === Train/test split ===
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# === Reshape data for LSTM ===
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# === Build LSTM model ===
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Define input shape using Input layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(1))  # Output layer for CO2 prediction

model.compile(optimizer='adam', loss='mean_squared_error')

# === Train the model ===
model.fit(X_train, y_train, epochs=20, batch_size=32)

# === Predict next week (336 steps = 7 days * 48 intervals/day) ===
forecast_steps = 336
current_input = df_scaled[features].values[-sequence_length:]
forecast_scaled = []

# Check the shape of current_input before reshaping
print("Shape of current_input:", current_input.shape)

for _ in range(forecast_steps):
    # Ensure the reshaping matches the expected shape (1, 48, 6)
    x_input = current_input.reshape(1, sequence_length, len(features) - 1)  # 48 time steps, 6 features
    y_pred = model.predict(x_input)[0][0]
    forecast_scaled.append(y_pred)

    # Prepare next input for next prediction
    new_row = current_input[-1].copy()
    new_row[features.index('CO2 (PPM)')] = y_pred  # Update CO2 value
    current_input = np.vstack([current_input[1:], new_row])

# === Inverse transform only CO2 ===
co2_index = features.index('CO2 (PPM)')
blank = np.zeros((forecast_steps, len(features)))
blank[:, co2_index] = forecast_scaled
forecast_co2 = scaler.inverse_transform(blank)[:, co2_index]

# === Generate timestamps for next 7 days ===
start_time = df['DateTime'].iloc[-1] + timedelta(minutes=30)
predicted_timestamps = [start_time + timedelta(minutes=30*i) for i in range(forecast_steps)]

# === Plot forecast ===
plt.figure(figsize=(14, 5))
plt.plot(predicted_timestamps, forecast_co2, label="Predicted CO₂ (ppm)", color='blue')
plt.axhline(y=800, color='red', linestyle='--', label='Ventilation Threshold')
plt.title("Predicted CO₂ Levels for the Next Week (30-minute intervals) using LSTM")
plt.xlabel("Date & Time")
plt.ylabel("CO₂ Level (ppm)")
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
