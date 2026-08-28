import yfinance as yf
import pandas as pd
import re


def get_historical_returns(portfolio, period="1y", end_date=None):
    tickers = portfolio.tickers()

    if end_date is None:
        end_date = portfolio.date

    end_date = pd.Timestamp(end_date)
    start_date = end_date - pd.DateOffset(years=get_period(period))

    prices = get_historical_prices(tickers, start_date, end_date)

    returns = prices.pct_change().dropna()

    if returns.empty:
        raise ValueError("Not enough price data to calculate historical returns")

    return returns


def get_historical_prices(tickers, start_date, end_date):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    prices = yf.download(
        tickers,
        start=start_date,
        end=end_date + pd.Timedelta(days=1),
        auto_adjust=True,
        progress=False
    )["Close"]

    if prices.empty:
        raise ValueError("No price data available.")

    return prices


def get_period(period="1y"):
    match = re.fullmatch(r"\d+y", period)

    if match:
        return int(period[:-1])

    raise ValueError(f"Unsupported period: {period}")
