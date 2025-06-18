"""
Markowitz Portfolio Optimization Model

This module implements the markowitz portfolio optimization model, which helps you build
an efficient frontier of optimal portfolios that give you the best expected return for
a given level of risk.

How it works:
1. Loads financial data from csv files or fetches from AlphaVantage API
2. Calculates expected returns and covariance matrix
3. Generates the efficient frontier
4. Calculates the capital allocation line and finds the tangency portfolio
5. Visualizes the results

@author: Pedro Gronda Garrigues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import requests
from datetime import datetime, timedelta
import time

class MarkowitzModel:
    """
    A class that implements the markowitz portfolio optimization model.
    Helps you find the best portfolio weights to maximize returns while keeping risk in check.
    """
    
    def __init__(self, risk_free_rate=0.02):
        """
        Init the markowitz model.
        
        Params:
        ------
        risk_free_rate : float, optional
            the risk-free interest rate (annual), defaults to 0.02 (2%)
        """
        self.risk_free_rate = risk_free_rate
        self.returns = None
        self.cov_matrix = None
        self.assets = None
        self.efficient_frontier = None
        self.optimal_portfolio = None
        
    def load_data_from_csv(self, directory_path, start_date=None, end_date=None):
        """
        Loads financial data from csv files in the specified directory.
        
        Params:
        ------
        directory_path : str
            Path to the dir containing csv files with financial data
        start_date : str, optional
            Start date for filtering data (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for filtering data (format: 'YYYY-MM-DD')
            
        Returns:
        -------
        pandas.DataFrame
            df containing the daily returns of each asset
        """

        csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"status: no csv files found in {directory_path}")
        
        all_data = {}
        for file in csv_files:
            asset_name = os.path.splitext(file)[0]
            file_path = os.path.join(directory_path, file)
            
            # date, open, high, low, close, volume
            df = pd.read_csv(file_path)
            
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            all_data[asset_name] = df['Close']
        
        prices_df = pd.DataFrame(all_data)
        
        returns_df = prices_df.pct_change().dropna()
        
        self.returns = returns_df
        self.assets = returns_df.columns.tolist()
        
        self.cov_matrix = returns_df.cov() * 252 # annualize
        
        return returns_df
    
    def fetch_data_from_alphavantage(self, symbols, api_key, time_period='5y'):
        """
        Fetch financial data from AlphaVantage API.
        
        Parameters:
        -----------
        symbols : list
            List of stock symbols to fetch data for
        api_key : str
            AlphaVantage API key
        time_period : str, optional
            Time period to fetch data for, by default '5y' (5 years)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the daily returns of each asset
        """
    
        all_data = {}
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            
            # AlphaVantage API endpoint for daily time series
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api_key}"
            
            response = requests.get(url)
            data = response.json()
            
            if "Error Message" in data:
                print(f"Error fetching data for {symbol}: {data['Error Message']}")
                continue
            
            if "Time Series (Daily)" not in data:
                print(f"No data available for {symbol}")
                continue
            
            time_series = data["Time Series (Daily)"]
            
            df = pd.DataFrame(time_series).T
            df.index = pd.to_datetime(df.index)
            df = df.apply(pd.to_numeric)
            df.rename(columns={"4. close": "Close"}, inplace=True)
            
            if time_period.endswith('y'):
                years = int(time_period[:-1])
                start_date = datetime.now() - timedelta(days=years*365)
                df = df[df.index >= start_date]
            elif time_period.endswith('m'):
                months = int(time_period[:-1])
                start_date = datetime.now() - timedelta(days=months*30)
                df = df[df.index >= start_date]
            
            all_data[symbol] = df["Close"]
            
            # AlphaVantage -> rate limit of 5 API calls per minute for free tier
            time.sleep(12)
        
        prices_df = pd.DataFrame(all_data)
        
        returns_df = prices_df.pct_change().dropna()
        
        self.assets = list(returns_df.columns)
        self.returns = returns_df
        
        self.cov_matrix = returns_df.cov() * 252  # assuming 252 trading days / year
        
        return returns_df
    
    def _portfolio_annualized_performance(self, weights):
        """
        Calculate annualized return and volatility for a portfolio.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Portfolio weights
            
        Returns:
        --------
        tuple
            (returns, volatility)
        """

        # expected returns (annualized)
        returns = np.sum(self.returns.mean() * weights) * 252
        
        # expected volatility (annualized)
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        return returns, volatility
    
    def _negative_sharpe_ratio(self, weights):
        """
        Calculate the negative Sharpe ratio of a portfolio.
        Used for optimization (minimizing negative Sharpe is maximizing Sharpe).
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Portfolio weights
            
        Returns:
        --------
        float
            Negative Sharpe ratio
        """
    
        returns, volatility = self._portfolio_annualized_performance(weights)
        sharpe_ratio = (returns - self.risk_free_rate) / volatility
        
        return -sharpe_ratio # negative Sharpe ratio for minimization!
    
    def _portfolio_volatility(self, weights):
        """
        Calculate the volatility of a portfolio.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Portfolio weights
            
        Returns:
        --------
        float
            Portfolio volatility
        """
    
        return self._portfolio_annualized_performance(weights)[1]
    
    def _portfolio_return(self, weights):
        """
        Calculate the return of a portfolio.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Portfolio weights
            
        Returns:
        --------
        float
            Portfolio return
        """
    
        return self._portfolio_annualized_performance(weights)[0]
    
    def generate_efficient_frontier(self, num_portfolios=100, min_return=None, max_return=None):
        """
        Generate the efficient frontier by optimizing portfolios for different target returns.
        
        Parameters:
        -----------
        num_portfolios : int, optional
            Number of portfolios to generate, by default 100
        min_return : float, optional
            Minimum target return, by default None
        max_return : float, optional
            Maximum target return, by default None
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the weights, returns, and volatilities of the efficient frontier portfolios
        """
    
        if self.returns is None or self.cov_matrix is None:
            raise ValueError("Data must be loaded before generating the efficient frontier")
        
        num_assets = len(self.assets)
        
        # maximum Sharpe ratio
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = np.array([1/num_assets] * num_assets)
        
        optimal_result = minimize(
            self._negative_sharpe_ratio,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = optimal_result['x']
        optimal_return, optimal_volatility = self._portfolio_annualized_performance(optimal_weights)
        optimal_sharpe = (optimal_return - self.risk_free_rate) / optimal_volatility
        
        self.optimal_portfolio = {
            'weights': dict(zip(self.assets, optimal_weights)),
            'return': optimal_return,
            'volatility': optimal_volatility,
            'sharpe_ratio': optimal_sharpe
        }
        
        min_vol_result = minimize(
            self._portfolio_volatility,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        min_vol_weights = min_vol_result['x']
        min_vol_return, min_vol_volatility = self._portfolio_annualized_performance(min_vol_weights)
        
        max_return_weights = np.zeros(num_assets)
        max_return_idx = np.argmax(self.returns.mean())
        max_return_weights[max_return_idx] = 1
        max_return_value, max_return_volatility = self._portfolio_annualized_performance(max_return_weights)
        
        if min_return is None:
            min_return = min_vol_return
        if max_return is None:
            max_return = max_return_value
        
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        efficient_weights = []
        efficient_returns = []
        efficient_volatilities = []
        
        for target in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self._portfolio_return(x) - target}
            )
            
            result = minimize(
                self._portfolio_volatility,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result['success']:
                weights = result['x']
                returns, volatility = self._portfolio_annualized_performance(weights)
                
                efficient_weights.append(dict(zip(self.assets, weights)))
                efficient_returns.append(returns)
                efficient_volatilities.append(volatility)
        
        self.efficient_frontier = pd.DataFrame({
            'returns': efficient_returns,
            'volatilities': efficient_volatilities,
            'weights': efficient_weights
        })
        
        return self.efficient_frontier
    
    def get_optimal_portfolio(self):
        """
        Get the optimal portfolio (maximum Sharpe ratio).
        
        Returns:
        --------
        dict
            Dictionary containing the optimal portfolio details
        """
        
        if self.optimal_portfolio is None:
            self.generate_efficient_frontier()
        
        return self.optimal_portfolio
    
    def get_min_volatility_portfolio(self):
        """
        Get the minimum volatility portfolio.
        
        Returns:
        --------
        dict
            Dictionary containing the minimum volatility portfolio details
        """
        
        if self.efficient_frontier is None:
            self.generate_efficient_frontier()
        
        min_vol_idx = self.efficient_frontier['volatilities'].idxmin()
        
        return {
            'weights': self.efficient_frontier.loc[min_vol_idx, 'weights'],
            'return': self.efficient_frontier.loc[min_vol_idx, 'returns'],
            'volatility': self.efficient_frontier.loc[min_vol_idx, 'volatilities'],
            'sharpe_ratio': (self.efficient_frontier.loc[min_vol_idx, 'returns'] - self.risk_free_rate) / self.efficient_frontier.loc[min_vol_idx, 'volatilities']
        }
    
    def get_max_return_portfolio(self):
        """
        Get the maximum return portfolio.
        
        Returns:
        --------
        dict
            Dictionary containing the maximum return portfolio details
        """
        
        if self.efficient_frontier is None:
            self.generate_efficient_frontier()
        
        max_return_idx = self.efficient_frontier['returns'].idxmax()
        
        return {
            'weights': self.efficient_frontier.loc[max_return_idx, 'weights'],
            'return': self.efficient_frontier.loc[max_return_idx, 'returns'],
            'volatility': self.efficient_frontier.loc[max_return_idx, 'volatilities'],
            'sharpe_ratio': (self.efficient_frontier.loc[max_return_idx, 'returns'] - self.risk_free_rate) / self.efficient_frontier.loc[max_return_idx, 'volatilities']
        }
    
    def get_target_return_portfolio(self, target_return):
        """
        Get a portfolio with a target return.
        
        Parameters:
        -----------
        target_return : float
            Target return
            
        Returns:
        --------
        dict
            Dictionary containing the portfolio details
        """
        
        if self.efficient_frontier is None:
            self.generate_efficient_frontier()
        
        idx = (self.efficient_frontier['returns'] - target_return).abs().idxmin()
        
        return {
            'weights': self.efficient_frontier.loc[idx, 'weights'],
            'return': self.efficient_frontier.loc[idx, 'returns'],
            'volatility': self.efficient_frontier.loc[idx, 'volatilities'],
            'sharpe_ratio': (self.efficient_frontier.loc[idx, 'returns'] - self.risk_free_rate) / self.efficient_frontier.loc[idx, 'volatilities']
        }
    
    def get_target_volatility_portfolio(self, target_volatility):
        """
        Get a portfolio with a target volatility.
        
        Parameters:
        -----------
        target_volatility : float
            Target volatility
            
        Returns:
        --------
        dict
            Dictionary containing the portfolio details
        """
        
        if self.efficient_frontier is None:
            self.generate_efficient_frontier()
        
        idx = (self.efficient_frontier['volatilities'] - target_volatility).abs().idxmin()
        
        return {
            'weights': self.efficient_frontier.loc[idx, 'weights'],
            'return': self.efficient_frontier.loc[idx, 'returns'],
            'volatility': self.efficient_frontier.loc[idx, 'volatilities'],
            'sharpe_ratio': (self.efficient_frontier.loc[idx, 'returns'] - self.risk_free_rate) / self.efficient_frontier.loc[idx, 'volatilities']
        }
    
    def plot_efficient_frontier(self, show_assets=True, show_cal=True, show_optimal=True, random_portfolios=0, figsize=(12, 8)):
        """
        Plot the efficient frontier.
        
        Parameters:
        -----------
        show_assets : bool, optional
            Whether to show individual assets, by default True
        show_cal : bool, optional
            Whether to show the capital allocation line, by default True
        show_optimal : bool, optional
            Whether to show the optimal portfolio, by default True
        random_portfolios : int, optional
            Number of random portfolios to generate, by default 0
        figsize : tuple, optional
            Figure size, by default (12, 8)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        
        if self.efficient_frontier is None:
            self.generate_efficient_frontier()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.efficient_frontier['volatilities'], self.efficient_frontier['returns'], 'b-', linewidth=3, label='Efficient Frontier')
        
        if show_assets:
            asset_returns = self.returns.mean() * 252
            asset_volatilities = self.returns.std() * np.sqrt(252)
            
            ax.scatter(asset_volatilities, asset_returns, marker='o', s=100, c='red', label='Assets')
            
            for i, asset in enumerate(self.assets):
                ax.annotate(asset, (asset_volatilities[i], asset_returns[i]), 
                           xytext=(10, 0), textcoords='offset points', fontsize=12)
        
        if show_cal and self.optimal_portfolio is not None:
            optimal_return = self.optimal_portfolio['return']
            optimal_volatility = self.optimal_portfolio['volatility']
            
            cal_slope = (optimal_return - self.risk_free_rate) / optimal_volatility
            
            x_min, x_max = ax.get_xlim()
            x = np.linspace(0, x_max * 1.2, 100)
            y = self.risk_free_rate + cal_slope * x
            
            ax.plot(x, y, 'g-', linewidth=2, label='Capital Allocation Line')
            
            ax.scatter(0, self.risk_free_rate, marker='*', s=200, c='green', label='Risk-Free Rate')
        
        if show_optimal and self.optimal_portfolio is not None:
            optimal_return = self.optimal_portfolio['return']
            optimal_volatility = self.optimal_portfolio['volatility']
            
            ax.scatter(optimal_volatility, optimal_return, marker='*', s=200, c='gold', 
                      edgecolors='black', label='Optimal Portfolio')
        
        if random_portfolios > 0:
            weights = np.random.random((random_portfolios, len(self.assets)))
            weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
            
            returns = np.zeros(random_portfolios)
            volatilities = np.zeros(random_portfolios)
            
            for i in range(random_portfolios):
                returns[i], volatilities[i] = self._portfolio_annualized_performance(weights[i])
            
            ax.scatter(volatilities, returns, marker='.', s=10, c='gray', alpha=0.5, label='Random Portfolios')
        
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Efficient Frontier')
        ax.grid(True)
        ax.legend()
        
        return fig