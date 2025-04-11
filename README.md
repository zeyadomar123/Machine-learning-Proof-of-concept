# Machine-learning-Proof-of-concept
# CO₂ Forecasting Using LSTM Neural Network

This Python script implements a time series prediction model for CO₂ concentration levels using an XGBoost. It is part of an indoor environmental monitoring system that forecasts CO₂ values over a 7-day period using historical sensor data collected at 30-minute intervals.

---

## Features

- Loads and preprocesses a CO₂ dataset (CSV format)
- Performs time-based feature engineering (hour, day, month, etc.)
- Normalizes the data using MinMaxScaler
- Trains an LSTM model using 24-hour sequences (48 time steps)
- Predicts CO₂ values for the next 7 days (336 steps)
- Plots forecasted values with a ventilation safety threshold line

---

## Dataset Requirements

- Input CSV file must contain at least the following columns:
  - `DateTime` (e.g. `"06/04 08:00"`)
  - `CO2 (PPM)`
  - `TVOC (PPB)`
- File path must be updated manually in the script:
  ```python
  df = pd.read_csv(r"C:\Users\zeyad\Documents\Well_Ventilated_CO2_Dataset.csv")
