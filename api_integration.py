"""
API Integration for Real-Time Data

This script demonstrates how to enhance the Markowitz model by integrating it with
external APIs to fetch real-time or near-real-time data for portfolio optimization.

It uses the Alpha Vantage API to fetch stock data and updates the model as new data
becomes available.

@author: Pedro Gronda Garrigues
"""

import os
import time
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from model import MarkowitzModel

class APIEnhancedMarkowitzModel(MarkowitzModel):
    """
    An enhanced version of the Markowitz model that integrates with external APIs
    to fetch real-time or near-real-time data for portfolio optimization.
    """
    
    def __init__(self, risk_free_rate=0.02, api_key=None):
        """
        Initialize the API-enhanced Markowitz model.
        
        Parameters:
        -----------
        risk_free_rate : float, optional
            The risk-free interest rate (annual), by default 0.02 (2%)
        api_key : str, optional
            Alpha Vantage API key, by default None
        """
        super().__init__(risk_free_rate)
        self.api_key = api_key
        self.last_update_time = None
        self.data_cache = {}
    
    def set_api_key(self, api_key):
        """
        Set the Alpha Vantage API key.
        
        Parameters:
        -----------
        api_key : str
            Alpha Vantage API key
        """
        self.api_key = api_key
    
    def fetch_daily_data(self, symbol, output_size='compact'):
        """
        Fetch daily data for a symbol from Alpha Vantage.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        output_size : str, optional
            Output size, by default 'compact'
            Options: 'compact' (latest 100 data points), 'full' (up to 20 years of data)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the daily data
        """
        if self.api_key is None:
            raise ValueError("API key must be set before fetching data")
        
        # Alpha Vantage API endpoint for daily data
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize={output_size}&apikey={self.api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            raise ValueError(f"Error fetching data for {symbol}: {data['Error Message']}")
        
        if "Time Series (Daily)" not in data:
            raise ValueError(f"No data available for {symbol}")
        
        time_series = data["Time Series (Daily)"]
        
        df = pd.DataFrame(time_series).T
        df.index = pd.to_datetime(df.index)
        df = df.apply(pd.to_numeric)
        df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adjusted Close",
            "6. volume": "Volume",
            "7. dividend amount": "Dividend",
            "8. split coefficient": "Split Coefficient"
        }, inplace=True)
        
        df.sort_index(inplace=True)  # Sort by date
        
        self.data_cache[symbol] = {
            'data': df,
            'last_update': datetime.now()
        }
        
        return df
    
    def fetch_multiple_symbols(self, symbols, output_size='compact'):
        """
        Fetch data for multiple symbols from Alpha Vantage.
        
        Parameters:
        -----------
        symbols : list
            List of stock symbols
        output_size : str, optional
            Output size, by default 'compact'
            
        Returns:
        --------
        dict
            Dictionary containing DataFrames for each symbol
        """
        all_data = {}
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            
            try:
                df = self.fetch_daily_data(symbol, output_size=output_size)
                all_data[symbol] = df
                
                # IMPORTANT: Alpha Vantage has a rate limit of 5 API calls per minute for free tier
                time.sleep(12)
            
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        return all_data
    
    def update_model_with_api_data(self, symbols, use_cache=True, cache_expiry_minutes=60, output_size='compact'):
        """
        Update the model with data fetched from Alpha Vantage API.
        
        Parameters:
        -----------
        symbols : list
            List of stock symbols
        use_cache : bool, optional
            Whether to use cached data if available, by default True
        cache_expiry_minutes : int, optional
            Cache expiry time in minutes, by default 60
        output_size : str, optional
            Output size, by default 'compact'
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the daily returns of each asset
        """
        all_data = {}
        
        for symbol in symbols:
            if (use_cache and symbol in self.data_cache and  # If cached data available
                (datetime.now() - self.data_cache[symbol]['last_update']).total_seconds() < cache_expiry_minutes * 60):
                print(f"Using cached data for {symbol}...")
                all_data[symbol] = self.data_cache[symbol]['data']
            else:
                print(f"Fetching data for {symbol}...")
                
                try:
                    df = self.fetch_daily_data(symbol, output_size=output_size)
                    all_data[symbol] = df
                    
                    # IMPORTANT: Alpha Vantage has a rate limit of 5 API calls per minute for free tier
                    time.sleep(12) 
                
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {e}")
                    continue
        
        # Extract closing prices
        prices = {}
        for symbol, df in all_data.items():
            if 'Adjusted Close' in df.columns:
                prices[symbol] = df['Adjusted Close']
            else:
                prices[symbol] = df['Close']
        
        # Combine all assets into a single DataFrame
        prices_df = pd.DataFrame(prices)
        returns_df = prices_df.pct_change().dropna()
        
        self.assets = list(returns_df.columns)
        self.returns = returns_df
        
        self.cov_matrix = returns_df.cov() * 252  # Assuming 252 trading days in a year
        
        self.last_update_time = datetime.now()
        
        return returns_df
    
    def schedule_regular_updates(self, symbols, update_interval_minutes=60, max_updates=None, output_size='compact'):
        """
        Schedule regular updates of the model with new data.
        
        Parameters:
        -----------
        symbols : list
            List of stock symbols
        update_interval_minutes : int, optional
            Update interval in minutes, by default 60
        max_updates : int, optional
            Maximum number of updates to perform, by default None (unlimited)
        output_size : str, optional
            Output size, by default 'compact'
            
        Returns:
        --------
        None
        """
        update_count = 0
        
        try:
            while max_updates is None or update_count < max_updates:
                print(f"\nUpdate {update_count + 1}" + (f" of {max_updates}" if max_updates else ""))
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                self.update_model_with_api_data(symbols, output_size=output_size)
                
                self.generate_efficient_frontier()
                
                optimal_portfolio = self.get_optimal_portfolio()
                
                print("\nOptimal Portfolio:")
                print(f"Expected Return: {optimal_portfolio['return']:.4f}")
                print(f"Volatility: {optimal_portfolio['volatility']:.4f}")
                print(f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.4f}")
                print("Weights:")
                for asset, weight in optimal_portfolio['weights'].items():
                    print(f"  {asset}: {weight:.4f}")
                
                update_count += 1
                
                if max_updates is None or update_count < max_updates:
                    print(f"\nWaiting {update_interval_minutes} minutes until next update...")
                    time.sleep(update_interval_minutes * 60)
        
        except KeyboardInterrupt:
            print("\nUpdates stopped by user.")
    
    def plot_real_time_efficient_frontier(self, show_assets=True, show_optimal=True, figsize=(12, 8)):
        """
        Plot the efficient frontier with real-time data.
        
        Parameters:
        -----------
        show_assets : bool, optional
            Whether to show individual assets, by default True
        show_optimal : bool, optional
            Whether to show the optimal portfolio, by default True
        figsize : tuple, optional
            Figure size, by default (12, 8)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        
        if self.efficient_frontier is None:
            self.generate_efficient_frontier()
        
        fig = self.plot_efficient_frontier(show_assets, show_optimal, figsize=figsize)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.axes[0].set_title(f'Efficient Frontier (as of {timestamp})')
        
        return fig
    
    def plot_price_history(self, figsize=(12, 8)):
        """
        Plot the price history of the assets.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size, by default (12, 8)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        
        if not self.data_cache:
            raise ValueError("No data available. Please fetch data first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for symbol, cache_item in self.data_cache.items():
            df = cache_item['data']
            
            normalized_prices = df['Close'] / df['Close'].iloc[0]
            
            ax.plot(df.index, normalized_prices, linewidth=2, label=symbol)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Price')
        ax.set_title('Price History')
        ax.grid(True)
        ax.legend()
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        return fig
    
    def plot_correlation_matrix(self, figsize=(10, 8)):
        """
        Plot the correlation matrix of the assets.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size, by default (10, 8)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        
        if self.returns is None:
            raise ValueError("No return data available. Please fetch data first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        corr_matrix = self.returns.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Correlation Matrix')
        
        return fig
