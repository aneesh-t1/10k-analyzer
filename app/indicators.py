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

# RSI
delta = data['Adj Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Plot
plt.figure(figsize=(12, 8))

# Price chart with moving averages
plt.subplot(2, 1, 1)
plt.plot(data['Adj Close'], label="Adj Close")
plt.plot(data['SMA_50'], label="50-day SMA", linestyle="--")
plt.plot(data['SMA_200'], label="200-day SMA", linestyle="--")
plt.title(f"{ticker} Price with Moving Averages")
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
