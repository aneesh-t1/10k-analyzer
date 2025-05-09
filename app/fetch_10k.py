import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate manual linear regression (no sklearn)
def linear_regression(X, y):
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean)**2)
    m = numerator / denominator
    b = y_mean - m * x_mean
    return m, b

# Function to predict using linear regression
def predict(m, b, x):
    return m * x + b

# Retrieve ticker data
ticker = "AAPL"
stock = yf.Ticker(ticker)
income_q = stock.quarterly_financials.T.sort_index(ascending=True).tail(10)

# List of indicators to predict
indicators = ["Total Revenue", "Net Income", "Free Cash Flow", "Operating Income"]

# Prepare figure for plotting
plt.figure(figsize=(12, 8))

# Loop through each indicator and apply linear regression
for idx, indicator in enumerate(indicators):
    if indicator in income_q.columns:
        indicator_series = income_q[indicator].dropna()
        indicator_series = indicator_series[::-1]  # Reverse to have the most recent first
        X = np.arange(len(indicator_series))  # [0, 1, ..., 9]
        y = indicator_series.values.astype(float)
        
        # Apply linear regression
        m, b = linear_regression(X, y)
        
        # Predict next value (next quarter)
        next_x = len(X)
        next_y = predict(m, b, next_x)

        # Plot actual data and predicted next quarter
        plt.subplot(len(indicators), 1, idx+1)
        plt.plot(indicator_series.index, y, marker='o', label=f"Actual {indicator}")
        plt.plot([indicator_series.index[-1], pd.Timestamp.now()], [y[-1], next_y], 'rx', label="Predicted Next Quarter")
        plt.plot(indicator_series.index.tolist() + [pd.Timestamp.now()], 
                 [predict(m, b, i) for i in range(len(X))] + [next_y],
                 linestyle='--', color='gray', label="Trend Line")
        plt.title(f"{ticker} {indicator} (Manual Linear Forecast)")
        plt.ylabel(indicator)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()

# Final layout
plt.tight_layout()
plt.show()

# Output predictions
for indicator in indicators:
    if indicator in income_q.columns:
        indicator_series = income_q[indicator].dropna()
        X = np.arange(len(indicator_series))  # [0, 1, ..., 9]
        y = indicator_series.values.astype(float)
        m, b = linear_regression(X, y)
        next_y = predict(m, b, len(X))
        print(f"\n📈 Predicted {indicator} (Next Quarter): ${next_y:,.0f}")
