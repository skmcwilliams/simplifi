# Simplifi

A comprehensive Python library for financial analysis and stock market data visualization. This library combines data from multiple sources including Yahoo Finance to provide a rich set of financial analysis tools.

## Features

- **Historical Data Analysis**
  - Fetch and process historical data from YahooQuery (Yahoo Finance)
  - Calculate average prices and logarithmic returns for BlackScholes model, along with moving averages for technical analysis
  - Option tog generate interactive OHLC (Open-High-Low-Close) candlestick chart including additional measures such as moving average and volume
  - Interactive buttons to select various time-based views and select/unselect data points
  - Returns historical dataframe with additional calculations

- **Options Analysis**
  - Black-Scholes option pricing model implementation
  - Prices entire options chain and returns pandas dataframe of prices

- **Dividend Discount Model**
  - Fetch stock valuation based on DDM calculations
  - Cost of Equity calculations
  - Risk-free rate analysis based on 10yr Treasury

**Get 10 Year**
  - Returns float of the 10-year treasury, a standard risk-free-rate
  - Cost of Equity calculations
  - Risk-free rate analysis based on 10yr Treasury

**Get CAPM Return**
  - Returns float expected return using Capital Asset Pricing Model, prints results and assumptions to terminal
  - Based on standard annual 8.5% target return, can be overriden using optional target_return parameter

## Installation

```bash
git clone https://github.com/skmcwilliams/simplifi
```

## Usage

### Basic Usage

```python
from simplifi import Simplifi

# Create an instance for a specific stock
stock = Simplifi('AAPL')

# Get historical data
historical_data_df = stock.get_historical_data()

# Get Black-Scholes option valuations
options_analysis_df = stock.blackscholes()

```

### Advanced Features

#### OHLC Chart Generation
```python
# Get historical data with OHLC chart
historical_data_df = stock.get_historical_data(make_ohlc=True)
```

Example OHLC chart output:

![OHLC Chart Example](chart.png)

#### Dividend Discount Model Valuation
```python
# Calculate stock valuation using DDM
ddm_valuation = stock.ddm_valuation()
```

## Dependencies

- numpy
- pandas
- plotly
- yahooquery
- beautifulsoup4
- requests
- scipy

## Data Sources
- Yahoo Finance (via yahooquery)

## Notes
- Features require internet connection for real-time data
- OHLC charts include 50-day and 200-day moving averages
- Black-Scholes calculations use the current 10-year Treasury rate as the risk-free rate

## License

See the LICENSE file for details.

## Contributing

Please feel free to submit a Pull Request.
