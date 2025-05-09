import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load ticker and data
ticker = "AAPL"
stock = yf.Ticker(ticker)

# Get quarterly financials (Income Statement) for the last 10 quarters
income_q = stock.quarterly_financials.T.sort_index(ascending=True).tail(10)

# Get quarterly balance sheet data
balance_sheet_q = stock.quarterly_balance_sheet.T

# Inspect columns to find the correct key for Stockholders Equity
print("Balance Sheet Columns:", balance_sheet_q.columns)

# Inspect the income statement columns to find the correct key for operating expenses
print("Income Statement Columns:", income_q.columns)

# Once you find the correct column for Operating Expenses (for example: 'Operating Income')
try:
    # Get Total Revenue from the income statement (drop missing values)
    revenue_series = income_q["Total Revenue"].dropna()
    revenue_series = revenue_series[::-1]  # Reverse so the most recent comes first

    # Get Operating Expenses (if found as 'Operating Income' or another name)
    expenses_series = income_q["Operating Income"].dropna()  # Update with correct column name if needed
    expenses_series = expenses_series[::-1]

    # Get Assets from the balance sheet (drop missing values)
    assets_series = balance_sheet_q["Total Assets"].dropna()
    assets_series = assets_series[::-1]

    # Get Stockholders Equity (SHE) from the balance sheet (drop missing values)
    she_series = balance_sheet_q["Stockholders Equity"].dropna()  # Corrected column name
    she_series = she_series[::-1]

    # Prepare X (time) and y (revenue)
    X = np.arange(len(revenue_series))  # [0, 1, ..., n]
    y = revenue_series.values.astype(float)

    # Manual linear regression for revenue prediction
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean)**2)
    m = numerator / denominator
    b = y_mean - m * x_mean

    # Predict next quarter revenue
    next_x = len(X)
    next_y = m * next_x + b

    # Plot revenue, expenses, assets, and stockholder equity
    plt.figure(figsize=(14, 8))

    # Plot revenue
    plt.subplot(2, 2, 1)
    plt.plot(revenue_series.index, y, marker='o', label="Actual Revenue")
    plt.plot([revenue_series.index[-1], pd.Timestamp.now()], [y[-1], next_y], 'rx', label="Predicted Next Quarter")
    plt.plot(revenue_series.index.tolist() + [pd.Timestamp.now()],
             [m * i + b for i in range(len(X))] + [next_y],
             linestyle='--', color='gray', label="Revenue Trend Line")
    plt.title(f"{ticker} Revenue Trend (Manual Linear Forecast)")
    plt.ylabel("Revenue ($)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Plot expenses
    plt.subplot(2, 2, 2)
    plt.plot(expenses_series.index, expenses_series.values, marker='o', label="Operating Expenses")
    plt.title(f"{ticker} Operating Expenses")
    plt.ylabel("Expenses ($)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Plot total assets
    plt.subplot(2, 2, 3)
    plt.plot(assets_series.index, assets_series.values, marker='o', label="Total Assets")
    plt.title(f"{ticker} Total Assets")
    plt.ylabel("Assets ($)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Plot shareholders equity (SHE)
    plt.subplot(2, 2, 4)
    plt.plot(she_series.index, she_series.values, marker='o', label="Stockholders Equity")
    plt.title(f"{ticker} Shareholders' Equity")
    plt.ylabel("Equity ($)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Output prediction
    print(f"\n📈 Predicted Revenue (Next Quarter): ${next_y:,.0f}")

except KeyError as e:
    print(f"KeyError: The key '{e.args[0]}' was not found in the data. Please inspect the balance sheet columns for the correct key.")
