import pandas as pd


class Portfolio:
    def __init__(self, date, positions):
        self.date = pd.Timestamp(date)
        self.positions = positions

    def tickers(self):
        return list(self.positions.keys())

    def weights(self):
        return self.positions

    def __repr__(self):
        return f"Portfolio(date={self.date.date()}, positions={self.positions})"


class PortfolioHistory:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])

    def dates(self):
        return sorted(self.df["date"].unique())

    def get_portfolio(self, date):
        date = pd.Timestamp(date)

        valid_dates = self.df.loc[self.df["date"] <= date, "date"]

        if valid_dates.empty:
            raise ValueError(f"No portfolio available on or before {date.date()}")

        portfolio_date = valid_dates.max()

        df_date = self.df[self.df["date"] == portfolio_date]

        positions = dict(zip(df_date["asset"], df_date["weight"]))

        return Portfolio(date=portfolio_date, positions=positions)

    def latest_portfolio(self):
        latest_date = self.df["date"].max()
        return self.get_portfolio(latest_date)


def load_portfolio_history(filepath="portfolio.csv"):
    df = pd.read_csv(filepath)

    required_columns = {"date", "asset", "weight"}

    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    return PortfolioHistory(df)
