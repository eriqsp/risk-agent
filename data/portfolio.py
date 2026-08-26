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


def load_portfolio(filename="portfolio.txt"):
    weights = {}

    with open(filename, "r") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split(',')

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid format on line {line_number}: "
                    f"'{line}'"
                )

            ticker = parts[0].strip().upper()
            weight_string = parts[1].strip()

            try:
                weight = float(weight_string)

            except ValueError:
                raise ValueError(
                    f"Invalid weight on line {line_number}: "
                    f"'{weight_string}'"
                )

            if weight < 0:
                raise ValueError(
                    f"Weight cannot be negative "
                    f"(line {line_number})."
                )

            if not ticker_exists(ticker):
                raise ValueError(
                    f"Invalid or unknown ticker '{ticker}' "
                    f"(line {line_number})."
                )

            if ticker in weights:
                raise ValueError(
                    f"Ticker '{ticker}' appears more than once."
                )

            weights[ticker] = weight

    return Portfolio(weights)
