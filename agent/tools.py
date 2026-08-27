import numpy as np
from langchain.tools import tool
from data.data import get_historical_returns


def create_tools(portfolio_history):
    allowed_periods = ["1y", "2y", "3y", "4y", "5y"]

    def resolve_portfolio(date: str | None):
        if date is None:
            return portfolio_history.latest_portfolio()

        return portfolio_history.get_portfolio(date)

    @tool
    def calculate_portfolio_volatility(date: str | None = None, period: str = "1y") -> float:
        """
        Calculate the annualized volatility of the portfolio as of a given date.

        Args:
            date:
                Portfolio evaluation date in YYYY-MM-DD format.
                The portfolio composition effective on or immediately
                before this date will be used.

            period:
                Historical lookback used to estimate volatility.
                Examples: "1y", "2y", "3y", "5y".
        """

        if period not in allowed_periods:
            raise ValueError(
                f"Invalid period '{period}'. "
                f"Choose from: {allowed_periods}"
            )

        try:
            portfolio = resolve_portfolio(date)
            returns = get_historical_returns(portfolio, period=period)

            weights = np.array([
                portfolio.positions[ticker]
                for ticker in returns.columns
            ])

            covariance_matrix = returns.cov()

            portfolio_variance = (
                weights
                @ covariance_matrix.values
                @ weights
            )

            volatility = np.sqrt(portfolio_variance)

            return float(volatility * np.sqrt(252))
        except Exception as e:
            raise RuntimeError(f"Unable to calculate portfolio volatility: {e}")

    @tool
    def calculate_asset_correlations(date: str | None = None):
        """
        Calculate the correlation matrix between the assets
        in the current portfolio using one year of daily returns.

        Args:
            date:
                Portfolio evaluation date in YYYY-MM-DD format.
                The portfolio composition effective on or immediately
                before this date will be used.
        """

        try:
            portfolio = resolve_portfolio(date)
            returns = get_historical_returns(portfolio)

            correlation = returns.corr()

            return correlation.to_dict()
        except Exception as e:
            raise RuntimeError(f"Unable to compute assets correlations: {e}")

    @tool
    def get_portfolio(date: str | None = None) -> dict[str, float]:
        """
        Return the current portfolio holdings and their weights.

        Args:
            date:
                Portfolio evaluation date in YYYY-MM-DD format.
                The portfolio composition effective on or immediately
                before this date will be used.
        """

        portfolio = resolve_portfolio(date)
        return portfolio.positions

    @tool
    def calculate_risk_contribution(date: str | None = None) -> dict[str, float]:
        """
        Calculate each asset's contribution to portfolio volatility
        using one year of daily historical returns.

        Args:
            date:
                Portfolio evaluation date in YYYY-MM-DD format.
                The portfolio composition effective on or immediately
                before this date will be used.
        """

        try:
            portfolio = resolve_portfolio(date)
            returns = get_historical_returns(portfolio)

            tickers = list(returns.columns)

            covariance_matrix = returns.cov()

            weights = np.array([
                portfolio.positions[ticker]
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
        except Exception as e:
            raise RuntimeError(f"Unable to calculate risk contribution: {e}")

    @tool
    def calculate_historical_var(date: str | None = None, confidence_level: float = 0.95, period: str = "1y") -> float:
        """
        Calculate the portfolio's 1-day historical Value at Risk.
        """

        if not 0 < confidence_level < 1:
            raise ValueError(
                "confidence_level must be between 0 and 1."
            )

        allowed_confidence_levels = [0.95, 0.975, 0.99]

        if confidence_level not in allowed_confidence_levels:
            raise ValueError(
                f"Invalid confidence level: {confidence_level}. "
                f"Choose from {allowed_confidence_levels}."
            )

        if period not in allowed_periods:
            raise ValueError(
                f"Invalid period: {period}. "
                f"Choose from {allowed_periods}."
            )

        try:
            portfolio = resolve_portfolio(date)
            returns = get_historical_returns(portfolio, period=period)

            weights = np.array([
                portfolio.positions[ticker]
                for ticker in returns.columns
            ])

            portfolio_returns = returns.values @ weights

            percentile = (1 - confidence_level) * 100

            var = -np.percentile(
                portfolio_returns,
                percentile
            )

            return float(var)
        except Exception as e:
            raise RuntimeError(f"Unable to calculate historical VaR: {e}")

    @tool
    def stress_test_portfolio(scenario: dict[str, float], date: str | None = None) -> float:
        """
        Calculate the portfolio return under a hypothetical stress scenario.

        Args:
            date:
                Portfolio evaluation date in YYYY-MM-DD format.
                The portfolio composition effective on or immediately
                before this date will be used.
            scenario: Dictionary mapping ticker symbols to hypothetical
                      returns. Returns must be expressed as decimals.

                      Assets not included in the scenario are assumed
                      to have a 0% return.

                      Example:
                      {
                          "AAPL": -0.20,
                          "MSFT": -0.10
                      }

        Returns:
            The hypothetical portfolio return.
        """

        portfolio = resolve_portfolio(date)
        unknown_assets = set(scenario) - set(portfolio.positions)

        if unknown_assets:
            raise ValueError(
                f"Scenario contains assets not in the portfolio: "
                f"{sorted(unknown_assets)}"
            )

        portfolio_return = sum(
            portfolio.positions[ticker] * scenario.get(ticker, 0.0)
            for ticker in portfolio.positions
        )

        return float(portfolio_return)

    @tool
    def calculate_max_drawdown(date: str | None = None, period: str = "1y") -> float:
        """
        Calculate the maximum drawdown of the portfolio.

        Args:
            date:
                Portfolio evaluation date in YYYY-MM-DD format.
                The portfolio composition effective on or immediately
                before this date will be used.

            period: Historical period. Valid values are
                    "1y", "2y", "3y", and "5y".
        """

        if period not in allowed_periods:
            raise ValueError(
                f"Invalid period: {period}. "
                f"Choose from {allowed_periods}."
            )

        try:
            portfolio = resolve_portfolio(date)
            returns = get_historical_returns(portfolio, period=period)

            weights = np.array([
                portfolio.positions[ticker]
                for ticker in returns.columns
            ])

            portfolio_returns = returns @ weights

            cumulative_returns = (1 + portfolio_returns).cumprod()

            running_max = cumulative_returns.cummax()

            drawdowns = (cumulative_returns / running_max) - 1

            max_drawdown = drawdowns.min()

            return float(max_drawdown)
        except Exception as e:
            raise RuntimeError(f"Unable to calculate maximum drawdown: {e}")

    return [
        get_portfolio,
        calculate_portfolio_volatility,
        calculate_asset_correlations,
        calculate_risk_contribution,
        calculate_historical_var,
        stress_test_portfolio,
        calculate_max_drawdown
    ]
