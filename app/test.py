import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get data (Disable auto_adjust to keep data simple)
ticker = "AAPL"
data = yf.download(ticker, start="2018-01-01", end="2025-01-01", auto_adjust=False)

# Moving Averages (50, 200)
data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()
data['SMA_200'] = data['Adj Close'].rolling(window=200).mean()

# Calculate RSI (Relative Strength Index)
delta = data['Adj Close'].diff()  # Calculate price difference
gain = delta.where(delta > 0, 0)  # Positive changes
loss = -delta.where(delta < 0, 0)  # Negative changes

# Calculate average gain and loss
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

# Calculate RSI
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Check if RSI column exists to avoid KeyError
print("RSI column exists:", 'RSI' in data.columns)

# Linear Regression for 50-day Moving Average (SMA_50)
x = np.arange(len(data['SMA_50']))
y = data['SMA_50'].dropna()  # Drop NaN values for SMA_50

# Perform linear regression (using numpy.polyfit)
slope, intercept = np.polyfit(x[-len(y):], y, 1)  # Fit the model to the last available data

# Predict the next year's values (for example, 252 trading days in a year)
future_x = np.arange(len(data['SMA_50']), len(data['SMA_50']) + 252)  # Next 252 days
predicted_sma_50 = slope * future_x + intercept  # Linear model for future prediction

# Plot
plt.figure(figsize=(12, 8))

# Price chart with moving averages and prediction
plt.subplot(2, 1, 1)
plt.plot(data['Adj Close'], label="Adj Close")
plt.plot(data['SMA_50'], label="50-day SMA", linestyle="--")
plt.plot(data.index[-1] + pd.to_timedelta(future_x - len(data['SMA_50']), 'D'), predicted_sma_50, label="Predicted 50-day SMA (1 Year)", linestyle=":", color='red')
plt.title(f"{ticker} Price with Moving Averages and Predicted Line")
plt.legend()

# RSI chart
plt.subplot(2, 1, 2)
plt.plot(data['RSI'], label="RSI", color='orange')
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.title(f"{ticker} RSI")
plt.legend()

plt.tight_layout()
plt.show()

print(f"Slope of the predicted 50-day SMA line: {slope}")

