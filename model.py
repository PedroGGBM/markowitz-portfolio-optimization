"""
Markowitz Portfolio Optimization Model

This module implements the basic Markowitz portfolio optimization model, which helps you build
an efficient frontier of optimal portfolios that give you the best expected return for
a given level of risk.

How it works:
1. Loads financial data from csv files
2. Calculates expected returns and covariance matrix
3. Generates the efficient frontier
4. Finds the optimal portfolio

@author: Pedro Gronda Garrigues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

class MarkowitzModel:
    """
    A class that implements the Markowitz portfolio optimization model.
    Helps you find the best portfolio weights to maximize returns while keeping risk in check.
    """
    
    def __init__(self, risk_free_rate=0.02):
        """
        Initialize the Markowitz model.
        
        Parameters:
        -----------
        risk_free_rate : float, optional
            The risk-free interest rate (annual), by default 0.02 (2%)
        """
        self.risk_free_rate = risk_free_rate
        self.returns = None
        self.cov_matrix = None
        self.assets = None
        self.efficient_frontier = None
        self.optimal_portfolio = None
        
    def load_data_from_csv(self, directory_path, start_date=None, end_date=None):
        """
        Load financial data from CSV files in the specified directory.
        
        Parameters:
        -----------
        directory_path : str
            Path to the directory containing CSV files with financial data
        start_date : str, optional
            Start date for filtering data (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for filtering data (format: 'YYYY-MM-DD')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the daily returns of each asset
        """
        csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {directory_path}")
        
        all_data = {}
        for file in csv_files:
            asset_name = os.path.splitext(file)[0]
            file_path = os.path.join(directory_path, file)
            
            # Read CSV file (expected format: Date, Open, High, Low, Close, Volume)
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
        
        # Annualize the covariance matrix (assuming 252 trading days in a year)
        self.cov_matrix = returns_df.cov() * 252
        
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
        # Expected returns (annualized)
        returns = np.sum(self.returns.mean() * weights) * 252
        
        # Expected volatility (annualized)
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
        
        return -sharpe_ratio
    
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
    
    def generate_efficient_frontier(self, num_portfolios=100):
        """
        Generate the efficient frontier by optimizing portfolios for different target returns.
        
        Parameters:
        -----------
        num_portfolios : int, optional
            Number of portfolios to generate, by default 100
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the weights, returns, and volatilities of the efficient frontier portfolios
        """
        if self.returns is None or self.cov_matrix is None:
            raise ValueError("Data must be loaded before generating the efficient frontier")
        
        num_assets = len(self.assets)
        
        # Find the maximum Sharpe ratio portfolio
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
        
        # Find the minimum volatility portfolio
        min_vol_result = minimize(
            self._portfolio_volatility,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        min_vol_weights = min_vol_result['x']
        min_vol_return, min_vol_volatility = self._portfolio_annualized_performance(min_vol_weights)
        
        # Find the maximum return portfolio (100% in the asset with highest return)
        max_return_weights = np.zeros(num_assets)
        max_return_idx = np.argmax(self.returns.mean())
        max_return_weights[max_return_idx] = 1
        max_return_value, max_return_volatility = self._portfolio_annualized_performance(max_return_weights)
        
        # Generate the efficient frontier
        target_returns = np.linspace(min_vol_return, max_return_value, num_portfolios)
        
        efficient_weights = []
        efficient_returns = []
        efficient_volatilities = []
        
        for target in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self._portfolio_annualized_performance(x)[0] - target}
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
    
    def plot_efficient_frontier(self, show_assets=True, show_optimal=True, figsize=(10, 6)):
        """
        Plot the efficient frontier.
        
        Parameters:
        -----------
        show_assets : bool, optional
            Whether to show individual assets, by default True
        show_optimal : bool, optional
            Whether to show the optimal portfolio, by default True
        figsize : tuple, optional
            Figure size, by default (10, 6)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.efficient_frontier is None:
            self.generate_efficient_frontier()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the efficient frontier
        ax.plot(self.efficient_frontier['volatilities'], self.efficient_frontier['returns'], 'b-', linewidth=3, label='Efficient Frontier')
        
        if show_assets:
            # Plot individual assets
            asset_returns = self.returns.mean() * 252
            asset_volatilities = self.returns.std() * np.sqrt(252)
            
            ax.scatter(asset_volatilities, asset_returns, marker='o', s=100, c='red', label='Assets')
            
            # Add asset labels
            for i, asset in enumerate(self.assets):
                ax.annotate(asset, (asset_volatilities[i], asset_returns[i]), 
                           xytext=(10, 0), textcoords='offset points', fontsize=10)
        
        if show_optimal and self.optimal_portfolio is not None:
            # Plot the optimal portfolio
            optimal_return = self.optimal_portfolio['return']
            optimal_volatility = self.optimal_portfolio['volatility']
            
            ax.scatter(optimal_volatility, optimal_return, marker='*', s=200, c='gold', 
                      edgecolors='black', label='Optimal Portfolio')
        
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Efficient Frontier')
        ax.grid(True)
        ax.legend()
        
        return fig
