import numpy as np
import yfinance as yf
from langchain.tools import tool
from data import portfolio


@tool
def calculate_portfolio_volatility():
    """
    Calculate the annualized volatility of the current portfolio
    using one year of daily historical prices.
    """

    tickers = list(portfolio.keys())

    # Get historical prices
    prices = yf.download(
        tickers,
        period="1y",
        auto_adjust=True,
        progress=False
    )["Close"]

    # Calculate daily asset returns
    returns = prices.pct_change().dropna()

    # Calculate portfolio returns
    portfolio_returns = sum(
        returns[ticker] * weight
        for ticker, weight in portfolio.items()
    )

    # Calculate annualized volatility
    daily_volatility = np.std(
        portfolio_returns,
        ddof=1
    )

    annualized_volatility = daily_volatility * np.sqrt(252)

    return float(annualized_volatility)


@tool
def calculate_asset_correlations():
    """
    Calculate the correlation matrix between the assets
    in the current portfolio using one year of daily returns.
    """

    tickers = list(portfolio.keys())

    prices = yf.download(
        tickers,
        period="1y",
        auto_adjust=True,
        progress=False
    )["Close"]

    returns = prices.pct_change().dropna()

    correlation = returns.corr()

    return correlation.to_dict()


@tool
def stress_test(shocks: dict[str, float]) -> float:
    """
    Calculate the percentage impact on the portfolio given
    hypothetical percentage price changes for each asset.

    shocks should contain asset tickers as keys and percentage
    changes as decimal values.

    Example:
    {"AAPL": -0.20, "MSFT": -0.10, "GOOG": -0.15}
    """

    portfolio_return = 0.0

    for ticker, shock in shocks.items():

        if ticker not in portfolio:
            raise ValueError(
                f"{ticker} is not in the portfolio."
            )

        portfolio_return += portfolio[ticker] * shock

    return portfolio_return


if __name__ == "__main__":

    result = stress_test.invoke({
        "shocks": {
            "AAPL": -0.20,
            "MSFT": -0.10,
            "GOOG": -0.15
        }
    })

    print(result)