"""
Machine Learning Enhancement

This script demonstrates how to enhance the Markowitz model with machine learning
techniques to predict future returns and volatility.

It uses:
1. ARIMA models for time series forecasting of returns
2. GARCH models for volatility forecasting
3. LSTM neural networks as an alternative approach

@author: Pedro Gronda Garrigues
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
from model import MarkowitzModel

warnings.filterwarnings('ignore')

class MLEnhancedMarkowitzModel(MarkowitzModel):
    """
    An enhanced version of the Markowitz model that uses machine learning
    to predict future returns and volatility.
    """
    
    def __init__(self, risk_free_rate=0.02):
        """
        Initialize the ML-enhanced Markowitz model.
        
        Parameters:
        -----------
        risk_free_rate : float, optional
            The risk-free interest rate (annual), by default 0.02 (2%)
        """
        super().__init__(risk_free_rate)
        self.predicted_returns = None
        self.predicted_cov_matrix = None
        
    def predict_returns_arima(self, forecast_periods=30):
        """
        Predict future returns using ARIMA models.
        
        Parameters:
        -----------
        forecast_periods : int, optional
            Number of periods to forecast, by default 30
            
        Returns:
        --------
        pandas.Series
            Predicted annualized returns for each asset
        """
        if self.returns is None:
            raise ValueError("Data must be loaded before predicting returns")
        
        predicted_returns = {}
        
        for asset in self.assets:
            print(f"Fitting ARIMA model for {asset}...")
            
            asset_returns = self.returns[asset]
            
            # fit ARIMA model
            # using a simple ARIMA(1,0,1) model for demonstration
            # in practice -> use auto_arima or grid search to find the best parameters
            model = ARIMA(asset_returns, order=(1, 0, 1))
            model_fit = model.fit()
            
            forecast = model_fit.forecast(steps=forecast_periods)
            
            predicted_return = forecast.mean() * 252
            predicted_returns[asset] = predicted_return
        
        # Store predicted returns
        self.predicted_returns = pd.Series(predicted_returns)
        
        return self.predicted_returns
    
    def predict_volatility_garch(self, forecast_periods=30):
        """
        Predict future volatility using GARCH models.
        
        Parameters:
        -----------
        forecast_periods : int, optional
            Number of periods to forecast, by default 30
            
        Returns:
        --------
        pandas.DataFrame
            Predicted annualized covariance matrix
        """
        if self.returns is None:
            raise ValueError("Data must be loaded before predicting volatility")
        
        num_assets = len(self.assets)
        predicted_cov = np.zeros((num_assets, num_assets))
        
        for i, asset in enumerate(self.assets):
            print(f"Fitting GARCH model for {asset}...")
            
            asset_returns = self.returns[asset].values
            
            # fit GARCH model
            # using a simple GARCH(1,1) model (for demonstration)
            model = arch_model(asset_returns, vol='Garch', p=1, q=1)
            model_fit = model.fit(disp='off')
            
            forecast = model_fit.forecast(horizon=forecast_periods)
            
            forecasted_var = forecast.variance.iloc[-1].mean()
            
            predicted_volatility = np.sqrt(forecasted_var * 252)
            
            predicted_cov[i, i] = predicted_volatility ** 2
        
        # for simplicity, use correlation from historical data
        # and combine it with the predicted volatilities to get the covariance matrix
        # in practice -> use a more sophisticated approach to predict the correlation matrix
        
        historical_corr = self.returns.corr().values
        
        predicted_std = np.sqrt(np.diag(predicted_cov))
        
        for i in range(num_assets):
            for j in range(num_assets):
                if i != j:
                    predicted_cov[i, j] = historical_corr[i, j] * predicted_std[i] * predicted_std[j]
        
        self.predicted_cov_matrix = pd.DataFrame(
            predicted_cov, 
            index=self.assets, 
            columns=self.assets
        )
        
        return self.predicted_cov_matrix
    
    def predict_returns_lstm(self, forecast_periods=30, epochs=50, batch_size=32, sequence_length=60):
        """
        Predict future returns using LSTM neural networks.
        
        Parameters:
        -----------
        forecast_periods : int, optional
            Number of periods to forecast, by default 30
        epochs : int, optional
            Number of training epochs, by default 50
        batch_size : int, optional
            Batch size for training, by default 32
        sequence_length : int, optional
            Length of input sequences, by default 60
            
        Returns:
        --------
        pandas.Series
            Predicted annualized returns for each asset
        """
        if self.returns is None:
            raise ValueError("Data must be loaded before predicting returns")
        
        predicted_returns = {}
        
        for asset in self.assets:
            print(f"Training LSTM model for {asset}...")
            
            asset_returns = self.returns[asset].values.reshape(-1, 1)
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(asset_returns)
            
            # sequences for training
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)
            
            # reshape X to be [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # LSTM model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            
            model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
            
            last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
            future_returns = []
            
            for _ in range(forecast_periods):
                next_return = model.predict(last_sequence, verbose=0)[0, 0]
                future_returns.append(next_return)
                
                last_sequence = np.append(last_sequence[:, 1:, :], [[next_return]], axis=1)
            
            future_returns = np.array(future_returns).reshape(-1, 1)
            future_returns = scaler.inverse_transform(future_returns)
            
            predicted_return = future_returns.mean() * 252
            predicted_returns[asset] = predicted_return[0]
        
        self.predicted_returns = pd.Series(predicted_returns)
        
        return self.predicted_returns
    
    def generate_efficient_frontier_with_predictions(self, num_portfolios=100, min_return=None, max_return=None):
        """
        Generate the efficient frontier using predicted returns and covariance matrix.
        
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
        
        if self.predicted_returns is None or self.predicted_cov_matrix is None:
            raise ValueError("Predictions must be made before generating the efficient frontier")
        
        # Store the original returns and covariance matrix
        original_returns = self.returns
        original_cov_matrix = self.cov_matrix
        
        # Replace with predicted values
        self.returns = pd.DataFrame(
            np.tile(self.predicted_returns.values, (len(original_returns), 1)) / 252,
            columns=self.assets,
            index=original_returns.index
        )
        
        self.cov_matrix = self.predicted_cov_matrix
        
        efficient_frontier = super().generate_efficient_frontier(num_portfolios, min_return, max_return)
        
        self.returns = original_returns
        self.cov_matrix = original_cov_matrix
        
        return efficient_frontier
    
    def compare_predictions_with_historical(self):
        """
        Compare predicted returns and volatilities with historical values.
        
        Returns:
        --------
        tuple
            (returns_comparison, volatility_comparison) DataFrames
        """
        
        if self.predicted_returns is None or self.predicted_cov_matrix is None:
            raise ValueError("Predictions must be made before comparing with historical values")
        
        historical_returns = self.returns.mean() * 252
        
        historical_volatilities = self.returns.std() * np.sqrt(252)
        
        predicted_volatilities = pd.Series(
            np.sqrt(np.diag(self.predicted_cov_matrix)),
            index=self.assets
        )
        
        returns_comparison = pd.DataFrame({
            'Historical': historical_returns,
            'Predicted': self.predicted_returns,
            'Difference': self.predicted_returns - historical_returns,
            'Percent Change': (self.predicted_returns - historical_returns) / historical_returns * 100
        })
        
        volatility_comparison = pd.DataFrame({
            'Historical': historical_volatilities,
            'Predicted': predicted_volatilities,
            'Difference': predicted_volatilities - historical_volatilities,
            'Percent Change': (predicted_volatilities - historical_volatilities) / historical_volatilities * 100
        })
        
        return returns_comparison, volatility_comparison
    
    def plot_comparison(self, figsize=(12, 10)):
        """
        Plot a comparison of historical and predicted returns and volatilities.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size, by default (12, 10)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        
        returns_comparison, volatility_comparison = self.compare_predictions_with_historical()
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        ax = axes[0]
        
        x = np.arange(len(self.assets))
        width = 0.35
        
        ax.bar(x - width/2, returns_comparison['Historical'], width, label='Historical')
        ax.bar(x + width/2, returns_comparison['Predicted'], width, label='Predicted')
        
        ax.set_xlabel('Asset')
        ax.set_ylabel('Expected Return')
        ax.set_title('Historical vs Predicted Returns')
        ax.set_xticks(x)
        ax.set_xticklabels(self.assets)
        ax.grid(True)
        ax.legend()
        
        ax = axes[1]
        
        ax.bar(x - width/2, volatility_comparison['Historical'], width, label='Historical')
        ax.bar(x + width/2, volatility_comparison['Predicted'], width, label='Predicted')
        
        ax.set_xlabel('Asset')
        ax.set_ylabel('Volatility')
        ax.set_title('Historical vs Predicted Volatilities')
        ax.set_xticks(x)
        ax.set_xticklabels(self.assets)
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        
        return fig
    
    def plot_efficient_frontiers_comparison(self, num_portfolios=100, figsize=(12, 8)):
        """
        Plot a comparison of the efficient frontiers using historical and predicted values.
        
        Parameters:
        -----------
        num_portfolios : int, optional
            Number of portfolios to generate, by default 100
        figsize : tuple, optional
            Figure size, by default (12, 8)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        
        if self.predicted_returns is None or self.predicted_cov_matrix is None:
            raise ValueError("Predictions must be made before comparing efficient frontiers")
        
        historical_frontier = super().generate_efficient_frontier(num_portfolios=num_portfolios)
        
        predicted_frontier = self.generate_efficient_frontier_with_predictions(num_portfolios=num_portfolios)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(
            historical_frontier['volatilities'],
            historical_frontier['returns'],
            'b-', linewidth=3,
            label='Historical Efficient Frontier'
        )
        
        ax.plot(
            predicted_frontier['volatilities'],
            predicted_frontier['returns'],
            'r-', linewidth=3,
            label='Predicted Efficient Frontier'
        )
        
        asset_returns = self.returns.mean() * 252
        asset_volatilities = self.returns.std() * np.sqrt(252)
        
        ax.scatter(asset_volatilities, asset_returns, marker='o', s=100, c='blue', label='Historical Assets')
        
        predicted_volatilities = np.sqrt(np.diag(self.predicted_cov_matrix))
        
        ax.scatter(
            predicted_volatilities,
            self.predicted_returns,
            marker='x', s=100, c='red',
            label='Predicted Assets'
        )
        
        for i, asset in enumerate(self.assets):
            ax.annotate(
                asset, 
                (asset_volatilities[i], asset_returns[i]),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=10,
                color='blue'
            )
            
            ax.annotate(
                asset,
                (predicted_volatilities[i], self.predicted_returns[i]),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=10,
                color='red'
            )
        
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Historical vs Predicted Efficient Frontiers')
        ax.grid(True)
        ax.legend()
        
        return fig