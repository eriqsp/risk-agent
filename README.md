### Structure

- portfolio.py 
  - Understands portfolio positions and portfolio dates

- data.py 
  - Retrieves market data for assets and historical periods

- tools.py
  - Exposes financial capabilities to the LLM

- agent.py
  - Defines model + tool-calling behavior

- main.py
  - Handles conversation/session loop

- portfolio.csv
  - Stores historical portfolio states


### Risk Tools

- Volatility, VaR, risk contribution, correlations, stress test: analyze one portfolio snapshot as of a date
- Maximum drawdown: follow the portfolio through time, using every weight change in portfolio.csv