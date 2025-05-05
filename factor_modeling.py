"""
Factor Modeling Enhancement

This script demonstrates how to enhance the Markowitz model with factor modeling
techniques, specifically the Fama-French three-factor model.

The Fama-French model extends the Capital Asset Pricing Model (CAPM) by adding
size risk and value risk factors to the market risk factor.

@author: Pedro Gronda Garrigues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
from model import MarkowitzModel

warnings.filterwarnings('ignore')

class FactorEnhancedMarkowitzModel(MarkowitzModel):
    """
    An enhanced version of the Markowitz model that uses factor modeling
    to estimate expected returns and risk.
    """
    
    def __init__(self, risk_free_rate=0.02):
        """
        Initialize the factor-enhanced Markowitz model.
        
        Parameters:
        -----------
        risk_free_rate : float, optional
            The risk-free interest rate (annual), by default 0.02 (2%)
        """
        super().__init__(risk_free_rate)
        self.factor_returns = None
        self.factor_loadings = None
        self.factor_cov_matrix = None
        self.idiosyncratic_var = None
        self.factor_expected_returns = None
        
    def generate_synthetic_factor_data(self, start_date=None, end_date=None):
        """
        Generate synthetic factor data for demonstration purposes.
        
        Parameters:
        -----------
        start_date : str, optional
            Start date for the synthetic data (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for the synthetic data (format: 'YYYY-MM-DD')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the synthetic factor returns
        """
        if self.returns is None:
            raise ValueError("Asset returns must be loaded before generating synthetic factor data")
        
        if start_date is None and end_date is None:
            dates = self.returns.index
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            dates = dates[dates.isin(self.returns.index)]
        
        # Generate synthetic factor returns
        np.random.seed(42)  # For reproducibility
        
        # Market factor (correlated with average market return)
        market_factor = self.returns.mean(axis=1) + np.random.normal(0, 0.005, len(dates))
        
        # Size factor (SMB - Small Minus Big)
        smb_factor = np.random.normal(0.001, 0.01, len(dates))
        
        # Value factor (HML - High Minus Low)
        hml_factor = np.random.normal(0.002, 0.012, len(dates))
        
        factor_data = pd.DataFrame({
            'MKT': market_factor,
            'SMB': smb_factor,
            'HML': hml_factor
        }, index=dates)
        
        self.factor_returns = factor_data
        
        return factor_data
    
    def estimate_factor_loadings(self, factors=None):
        """
        Estimate factor loadings (betas) for each asset using linear regression.
        
        Parameters:
        -----------
        factors : list, optional
            List of factor names to use, by default None (use all factors)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the factor loadings for each asset
        """
        if self.returns is None or self.factor_returns is None:
            raise ValueError("Asset returns and factor returns must be loaded before estimating factor loadings")
        
        if factors is None:
            factors = self.factor_returns.columns
        
        common_dates = self.returns.index.intersection(self.factor_returns.index)
        asset_returns = self.returns.loc[common_dates]
        factor_returns = self.factor_returns.loc[common_dates, factors]
        
        X = sm.add_constant(factor_returns)
        
        factor_loadings = pd.DataFrame(index=self.assets, columns=['alpha'] + list(factors))
        
        idiosyncratic_var = np.zeros(len(self.assets))
        
        for i, asset in enumerate(self.assets):
            model = sm.OLS(asset_returns[asset], X)
            results = model.fit()
            
            factor_loadings.loc[asset, 'alpha'] = results.params['const']
            for factor in factors:
                factor_loadings.loc[asset, factor] = results.params[factor]
            
            idiosyncratic_var[i] = results.mse_resid
        
        self.factor_loadings = factor_loadings
        self.idiosyncratic_var = idiosyncratic_var
        
        return factor_loadings
    
    def estimate_factor_expected_returns(self, method='historical'):
        """
        Estimate expected returns for the factors.
        
        Parameters:
        -----------
        method : str, optional
            Method to estimate factor expected returns, by default 'historical'
            Options: 'historical', 'risk_premium'
            
        Returns:
        --------
        pandas.Series
            Series containing the expected returns for each factor
        """
        if self.factor_returns is None:
            raise ValueError("Factor returns must be loaded before estimating factor expected returns")
        
        if method == 'historical':
            factor_expected_returns = self.factor_returns.mean() * 252
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.factor_expected_returns = factor_expected_returns
        
        return factor_expected_returns
    
    def estimate_factor_cov_matrix(self):
        """
        Estimate the covariance matrix of the factors.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the covariance matrix of the factors
        """
        if self.factor_returns is None:
            raise ValueError("Factor returns must be loaded before estimating factor covariance matrix")
        
        factor_cov_matrix = self.factor_returns.cov() * 252
        
        self.factor_cov_matrix = factor_cov_matrix
        
        return factor_cov_matrix
    
    def estimate_returns_with_factors(self):
        """
        Estimate expected returns for each asset using the factor model.
        
        Returns:
        --------
        pandas.Series
            Series containing the expected returns for each asset
        """
        if self.factor_loadings is None or self.factor_expected_returns is None:
            raise ValueError("Factor loadings and expected returns must be estimated first")
        
        # E[R_i] = alpha_i + sum(beta_i,j * E[F_j])
        expected_returns = pd.Series(index=self.assets)
        
        for asset in self.assets:
            # Start with alpha
            expected_return = self.factor_loadings.loc[asset, 'alpha']
            
            # Add factor contributions
            for factor in self.factor_expected_returns.index:
                if factor in self.factor_loadings.columns:
                    expected_return += self.factor_loadings.loc[asset, factor] * self.factor_expected_returns[factor]
            
            expected_returns[asset] = expected_return
        
        return expected_returns
    
    def estimate_cov_matrix_with_factors(self):
        """
        Estimate the covariance matrix of asset returns using the factor model.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the covariance matrix of asset returns
        """
        if (self.factor_loadings is None or self.factor_cov_matrix is None or 
            self.idiosyncratic_var is None):
            raise ValueError("Factor loadings, covariance matrix, and idiosyncratic variance must be estimated first")
        
        factor_loadings = self.factor_loadings.drop('alpha', axis=1).values
        
        # Cov_systematic = B * Cov_factors * B^T (systematic risk component)
        systematic_cov = factor_loadings @ self.factor_cov_matrix.values @ factor_loadings.T
        
        # Cov = Cov_systematic + diag(idiosyncratic_var) (idiosyncratic risk component)
        cov_matrix = systematic_cov + np.diag(self.idiosyncratic_var)
        
        cov_matrix_df = pd.DataFrame(cov_matrix, index=self.assets, columns=self.assets)
        
        return cov_matrix_df
    
    def generate_efficient_frontier_with_factors(self, num_portfolios=100):
        """
        Generate the efficient frontier using factor-based expected returns and covariance matrix.
        
        Parameters:
        -----------
        num_portfolios : int, optional
            Number of portfolios to generate, by default 100
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the weights, returns, and volatilities of the efficient frontier portfolios
        """
        expected_returns = self.estimate_returns_with_factors()
        cov_matrix = self.estimate_cov_matrix_with_factors()
        
        original_returns = self.returns
        original_cov_matrix = self.cov_matrix
        
        # Replace with factor-based values
        self.returns = pd.DataFrame(
            np.tile(expected_returns.values, (len(original_returns), 1)) / 252,
            columns=self.assets,
            index=original_returns.index
        )
        
        self.cov_matrix = cov_matrix
        
        efficient_frontier = super().generate_efficient_frontier(num_portfolios)
        
        # Restore original values
        self.returns = original_returns
        self.cov_matrix = original_cov_matrix
        
        return efficient_frontier
    
    def analyze_factor_contributions(self):
        """
        Analyze the contribution of each factor to the expected returns of each asset.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the factor contributions for each asset
        """
        if self.factor_loadings is None or self.factor_expected_returns is None:
            raise ValueError("Factor loadings and expected returns must be estimated first")
        
        contributions = pd.DataFrame(index=self.assets)
        
        contributions['alpha'] = self.factor_loadings['alpha']
        
        for factor in self.factor_expected_returns.index:
            if factor in self.factor_loadings.columns:
                contributions[factor] = self.factor_loadings[factor] * self.factor_expected_returns[factor]
        
        contributions['total'] = contributions.sum(axis=1)
        
        return contributions
    
    def plot_factor_loadings(self, figsize=(12, 8)):
        """
        Plot the factor loadings for each asset.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size, by default (12, 8)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.factor_loadings is None:
            raise ValueError("Factor loadings must be estimated first")
        
        loadings_plot = self.factor_loadings.drop('alpha', axis=1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(loadings_plot, annot=True, cmap='coolwarm', center=0, ax=ax)
        
        ax.set_title('Factor Loadings')
        ax.set_xlabel('Factors')
        ax.set_ylabel('Assets')
        
        return fig
    
    def plot_factor_contributions(self, figsize=(12, 8)):
        """
        Plot the factor contributions to the expected returns of each asset.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size, by default (12, 8)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        contributions = self.analyze_factor_contributions()
        
        contributions_plot = contributions.drop('total', axis=1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        contributions_plot.plot(kind='bar', stacked=True, ax=ax)
        
        ax.set_title('Factor Contributions to Expected Returns')
        ax.set_xlabel('Assets')
        ax.set_ylabel('Expected Return')
        ax.grid(True)
        ax.legend(title='Factors')
        
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        return fig
    
    def plot_efficient_frontiers_comparison(self, num_portfolios=100, figsize=(12, 8)):
        """
        Plot a comparison of the efficient frontiers using historical and factor-based values.
        
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
        historical_frontier = super().generate_efficient_frontier(num_portfolios=num_portfolios)
        
        factor_frontier = self.generate_efficient_frontier_with_factors(num_portfolios=num_portfolios)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(
            historical_frontier['volatilities'],
            historical_frontier['returns'],
            'b-', linewidth=3,
            label='Historical Efficient Frontier'
        )
        
        ax.plot(
            factor_frontier['volatilities'],
            factor_frontier['returns'],
            'g-', linewidth=3,
            label='Factor-based Efficient Frontier'
        )
        
        asset_returns = self.returns.mean() * 252
        asset_volatilities = self.returns.std() * np.sqrt(252)
        
        ax.scatter(asset_volatilities, asset_returns, marker='o', s=100, c='blue', label='Historical Assets')
        
        factor_returns = self.estimate_returns_with_factors()
        factor_cov_matrix = self.estimate_cov_matrix_with_factors()
        factor_volatilities = np.sqrt(np.diag(factor_cov_matrix))
        
        ax.scatter(
            factor_volatilities,
            factor_returns,
            marker='x', s=100, c='green',
            label='Factor-based Assets'
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
                (factor_volatilities[i], factor_returns[i]),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=10,
                color='green'
            )
        
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Historical vs Factor-based Efficient Frontiers')
        ax.grid(True)
        ax.legend()
        
        return fig
