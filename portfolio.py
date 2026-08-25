import yfinance as yf


class Portfolio:
    def __init__(self, weights):
        if not weights:
            raise ValueError("Portfolio cannot be empty.")

        if any(weight < 0 for weight in weights.values()):
            raise ValueError("Portfolio weights cannot be negative.")

        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError(
                "Portfolio weights must sum to 100%."
            )

        self.weights = weights

    def tickers(self):
        return list(self.weights.keys())


def ticker_exists(ticker):
    try:

        data = yf.download(
            ticker,
            period="5d",
            progress=False,
            auto_adjust=True
        )

        return not data.empty

    except Exception:
        return False


def create_portfolio():
    print("Enter your portfolio.")
    print("Press Enter without a ticker when you are finished.\n")

    weights = {}
    while True:
        ticker = input(
            "Ticker (or press Enter to finish): "
        ).strip().upper()

        if ticker == "":
            break

        if not ticker_exists(ticker):
            print(
                f"Could not find historical data for '{ticker}'. "
                "Please enter a valid ticker.\n"
            )

            continue

        if ticker in weights:
            print(
                f"{ticker} has already been added.\n"
            )

            continue

        weight = 0
        while True:
            weight_input = input(
                f"Weight for {ticker} (%): "
            ).strip()

            try:
                weight = float(weight_input)

            except ValueError:
                print(
                    "Invalid weight. "
                    "Please enter a number, e.g. 25 or 25.5.\n"
                )

                continue

            if weight < 0:
                print(
                    "Weight cannot be negative.\n"
                )

                continue

            break

        weights[ticker] = weight / 100

    return Portfolio(weights)
