import yfinance as yf
import pandas as pd


def get_historical_returns(
    portfolio,
    period="1y"
):

    tickers = portfolio.tickers()

    prices = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False
    )["Close"]

    returns = prices.pct_change().dropna()

    return returns