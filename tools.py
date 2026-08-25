import numpy as np
from langchain.tools import tool
from data import get_historical_returns


def create_tools(portfolio):

    @tool
    def calculate_portfolio_volatility():
        """
        Calculate the annualized volatility of the current portfolio
        using one year of daily historical returns.
        """

        returns = get_historical_returns(portfolio)

        weights = np.array([
            portfolio.weights[ticker]
            for ticker in returns.columns
        ])

        covariance_matrix = returns.cov()

        portfolio_variance = (
            weights
            @ covariance_matrix.values
            @ weights
        )

        portfolio_volatility = np.sqrt(portfolio_variance)

        return float(portfolio_volatility * np.sqrt(252))


    @tool
    def calculate_asset_correlations():
        """
        Calculate the correlation matrix between the assets
        in the current portfolio using one year of daily returns.
        """

        returns = get_historical_returns(portfolio)

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

            if ticker not in portfolio.weights:
                raise ValueError(
                    f"{ticker} is not in the portfolio."
                )

            portfolio_return += portfolio.weights[ticker] * shock

        return portfolio_return


    @tool
    def get_portfolio() -> dict[str, float]:
        """
        Return the current portfolio holdings and their weights.
        """

        return portfolio.weights


    @tool
    def calculate_risk_contribution() -> dict[str, float]:
        """
        Calculate each asset's contribution to portfolio volatility
        using one year of daily historical returns.
        """

        returns = get_historical_returns(portfolio)

        tickers = list(returns.columns)

        covariance_matrix = returns.cov()

        weights = np.array([
            portfolio.weights[ticker]
            for ticker in tickers
        ])

        portfolio_variance = (
            weights
            @ covariance_matrix.values
            @ weights
        )

        portfolio_volatility = np.sqrt(portfolio_variance)

        marginal_contribution = (
            covariance_matrix.values @ weights
        ) / portfolio_volatility

        component_contribution = (
            weights * marginal_contribution
        )

        component_contribution *= np.sqrt(252)

        return {
            ticker: float(component_contribution[i])
            for i, ticker in enumerate(tickers)
        }


    @tool
    def calculate_historical_var(confidence_level: float = 0.95) -> float:
        """
        Calculate the portfolio's 1-day historical Value at Risk.
        """

        if not 0 < confidence_level < 1:
            raise ValueError(
                "confidence_level must be between 0 and 1."
            )

        returns = get_historical_returns(portfolio)

        weights = np.array([
            portfolio.weights[ticker]
            for ticker in returns.columns
        ])

        portfolio_returns = returns.values @ weights

        percentile = (1 - confidence_level) * 100

        var = -np.percentile(
            portfolio_returns,
            percentile
        )

        return float(var)

    return [
        get_portfolio,
        calculate_portfolio_volatility,
        calculate_asset_correlations,
        calculate_risk_contribution,
        calculate_historical_var,
        stress_test
    ]
