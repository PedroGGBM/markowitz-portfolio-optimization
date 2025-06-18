# Markowitz Portfolio Optimization

This project implements the Markowitz portfolio optimization model, which is used to construct an efficient frontier of optimal portfolios that offer the highest expected return for a given level of risk.

## Overview

The Markowitz model, also known as Modern Portfolio Theory (MPT), is a mathematical framework for assembling a portfolio of assets such that the expected return is maximized for a given level of risk. It was pioneered by Harry Markowitz in 1952 and later earned him the Nobel Prize in Economics.

Implementation includes:

- Loading financial data from CSV files or fetching from AlphaVantage API
- Calculating expected returns and covariance matrix
- Generating the efficient frontier
- Calculating the capital allocation line and finding the tangency portfolio
- Visualizing the results
- Multiple enhancements to the basic model:
  - Machine learning for risk and return prediction
  - Factor modeling with Fama-French factors
  - Advanced Monte Carlo simulations
  - Real-time data integration with APIs

@author: Pedro Gronda Garrigues
@email:  pgrondagarrigues@gmail.com | pgg6@st-andrews.ac.uk

## Installation

### Prerequisites

- Python 3.9+
- Required packages:

```bash
pip install numpy pandas matplotlib scipy requests statsmodels arch tensorflow scikit-learn seaborn
```

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd markowitz
```

2. (Optional) Get an Alpha Vantage API key:
   - Sign up for a free API key at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Create a file at `config/api_keys.json` with the following content:
   ```json
   {
     "alpha_vantage_api_key": "YOUR_API_KEY"
   }
   ```

## Project Structure

- `src/model.py`: Base Markowitz model implementation
- `src/download_data.py`: Script to download sample data from Yahoo Finance
- `src/example.py`: Example script demonstrating basic usage
- `src/ml_enhancement.py`: Machine learning enhancement for risk and return prediction
- `src/factor_modeling.py`: Factor modeling enhancement with Fama-French factors
- `src/monte_carlo_enhancement.py`: Advanced Monte Carlo simulation enhancement
- `src/api_integration.py`: API integration for real-time data
- `src/main.py`: Main script with command-line interface to access all features

## Usage

### Command-Line Interface

The project provides a command-line interface through `main.py` that allows you to access all the implemented features:

```bash
# basic usage with sample data
python src/main.py --sample

# using CSV files as data source
python src/main.py --csv --data-dir data

# using Alpha Vantage API as data source
python src/main.py --api --symbols AAPL MSFT GOOGL AMZN META --api-key YOUR_API_KEY

# using machine learning enhancement
python src/main.py --csv --ml

# using factor modeling enhancement
python src/main.py --csv --factor

# using Monte Carlo enhancement
python src/main.py --csv --monte-carlo

# using all enhancements
python src/main.py --csv --all

# get help
python src/main.py --help
```

### Downloading Sample Data

```bash
# download 5 years of data for selected (popular) stocks
python src/download_data.py
```

### Basic Usage

```python
from src.model import MarkowitzModel

# create a Markowitz model instance
model = MarkowitzModel(risk_free_rate=0.02)

# load data from CSV files
model.load_data_from_csv('data')

# generate efficient frontier
efficient_frontier = model.generate_efficient_frontier(num_portfolios=100)

# get optimal portfolio
optimal_portfolio = model.get_optimal_portfolio()

# print optimal portfolio details
print("Optimal Portfolio:")
print(f"Expected Return: {optimal_portfolio['return']:.4f}")
print(f"Volatility: {optimal_portfolio['volatility']:.4f}")
print(f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.4f}")
print("Weights:")
for asset, weight in optimal_portfolio['weights'].items():
    print(f"  {asset}: {weight:.4f}")

# plot the efficient frontier
fig = model.plot_efficient_frontier(show_assets=True, show_cal=True, show_optimal=True, random_portfolios=500)
fig.savefig('efficient_frontier.png')
```

### Machine Learning Enhancement

```python
from src.ml_enhancement import MLEnhancedMarkowitzModel

# create an ML-enhanced Markowitz model instance
model = MLEnhancedMarkowitzModel(risk_free_rate=0.02)

# load data from CSV files
model.load_data_from_csv('data')

# predict returns using ARIMA
predicted_returns = model.predict_returns_arima()

# predict volatility using GARCH
predicted_cov_matrix = model.predict_volatility_garch()

# compare predictions with historical values
returns_comparison, volatility_comparison = model.compare_predictions_with_historical()

# generate the efficient frontier with predicted values
predicted_frontier = model.generate_efficient_frontier_with_predictions()

# plot comparison
fig = model.plot_comparison()
fig.savefig('ml_comparison.png')

# plot efficient frontiers comparison
fig = model.plot_efficient_frontiers_comparison()
fig.savefig('ml_efficient_frontiers.png')
```

### Factor Modeling Enhancement

```python
from src.factor_modeling import FactorEnhancedMarkowitzModel

# create a factor-enhanced Markowitz model instance
model = FactorEnhancedMarkowitzModel(risk_free_rate=0.02)

# load data from CSV files
model.load_data_from_csv('data')

# generate synthetic factor data
factor_data = model.generate_synthetic_factor_data()

# estimate factor loadings
factor_loadings = model.estimate_factor_loadings()

# estimate factor expected returns
factor_expected_returns = model.estimate_factor_expected_returns()

# estimate factor covariance matrix
factor_cov_matrix = model.estimate_factor_cov_matrix()

# analyze factor contributions
contributions = model.analyze_factor_contributions()

# generate the efficient frontier with factor-based values
factor_frontier = model.generate_efficient_frontier_with_factors()

# plot factor loadings
fig = model.plot_factor_loadings()
fig.savefig('factor_loadings.png')

# plot factor contributions
fig = model.plot_factor_contributions()
fig.savefig('factor_contributions.png')

# plot efficient frontiers comparison
fig = model.plot_efficient_frontiers_comparison()
fig.savefig('factor_efficient_frontiers.png')
```

### Advanced Monte Carlo Simulations

```python
from src.monte_carlo_enhancement import MonteCarloEnhancedMarkowitzModel

# create a Monte Carlo-enhanced Markowitz model instance
model = MonteCarloEnhancedMarkowitzModel(risk_free_rate=0.02)

# load data from CSV files
model.load_data_from_csv('data')

# generate the efficient frontier
model.generate_efficient_frontier()

# run Monte Carlo simulations using different methods
comparison_results = model.compare_simulation_methods(
    time_horizon=252,  # 1 year
    num_simulations=1000,
    initial_value=1.0,
    seed=42
)

# calculate risk metrics
metrics = model.calculate_risk_metrics(comparison_results['gbm']['simulation'])

# plot simulation results
fig = model.plot_simulation_results(comparison_results['gbm']['simulation'])
fig.savefig('mc_simulation.png')

# plot return distribution
fig = model.plot_return_distribution(comparison_results['gbm']['simulation'])
fig.savefig('mc_return_distribution.png')

# plot drawdown distribution
fig = model.plot_drawdown_distribution(comparison_results['gbm']['simulation'])
fig.savefig('mc_drawdown_distribution.png')

# plot method comparison
fig, metrics_df = model.plot_method_comparison(comparison_results)
fig.savefig('mc_method_comparison.png')
```

### API Integration for Real-Time Data [EXAMPLE]

```python
from src.api_integration import APIEnhancedMarkowitzModel

# create an API-enhanced Markowitz model instance
model = APIEnhancedMarkowitzModel(risk_free_rate=0.02, api_key='YOUR_API_KEY')

# update model with API data
model.update_model_with_api_data(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'], data_type='daily')

# generate the efficient frontier
model.generate_efficient_frontier()

# plot the efficient frontier with real-time data
fig = model.plot_real_time_efficient_frontier()
fig.savefig('api_efficient_frontier.png')

# plot price history
fig = model.plot_price_history()
fig.savefig('api_price_history.png')

# plot correlation matrix
fig = model.plot_correlation_matrix()
fig.savefig('api_correlation_matrix.png')

# schedule regular updates
model.schedule_regular_updates(
    ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'], # select stocks of choice
    update_interval_minutes=60,
    max_updates=24 # every hour for a day
)
```

## Data Format

The model expects CSV files with the following format:

```
Date,Open,High,Low,Close,Volume
2020-01-01,100.0,105.0,99.0,102.0,1000000
2020-01-02,102.0,106.0,101.0,105.0,1200000
...
```

Each CSV file should represent one asset, and the filename (without extension) will be used as the asset name.

## Key Concepts

```Source: Investopedia (the best)```

### Efficient Frontier

The efficient frontier is a set of optimal portfolios that offer the highest expected return for a given level of risk or the lowest risk for a given level of expected return.

### Capital Allocation Line (CAL)

The capital allocation line represents the combinations of the risk-free asset and the optimal risky portfolio (tangency portfolio). It is a straight line that passes through the risk-free rate on the y-axis and is tangent to the efficient frontier.

### Sharpe Ratio

The Sharpe ratio is a measure of the excess return (or risk premium) per unit of risk. It is calculated as:

```
Sharpe Ratio = (Expected Return - Risk-Free Rate) / Volatility
```

The optimal portfolio is the one with the highest Sharpe ratio, which is the tangency point between the capital allocation line and the efficient frontier.

### Factor Models

Factor models explain asset returns using a set of common risk factors. The Fama-French three-factor model extends the Capital Asset Pricing Model (CAPM) by adding size risk and value risk factors to the market risk factor.

### Monte Carlo Simulations

Monte Carlo simulations use random sampling to estimate the distribution of possible outcomes. In portfolio optimization, they are used to analyze the robustness of the portfolio across different market scenarios.

## Future Enhancements

- Web interface for interactive portfolio optimization
- Backtesting capabilities
- Support for constraints (e.g., sector allocation, ESG criteria)
- Integration with more data sources
- Optimization for specific investment goals (e.g., income, growth)

## TODO

- For ML enhancements: use auto_arima or grid search to find the best parameters
- For ML enhancements: improve correlation matrix prediction

## License

This project is licensed under the MIT License - see the LICENSE file for details.

DISCLAIMER: This project is fully open-source. If you wish to contribute, simply submit a Pull Request and I (the author) will review it a.s.a.p.!

Originally a project for Mercury Capital Management, University of St Andrews, project proposal. I then got bored during a plane ride and added enhancements...
