"""
Monte Carlo Simulation Enhancement

This script demonstrates how to enhance the Markowitz model with Monte Carlo
simulation techniques to analyze the robustness of the portfolio across different
market scenarios and calculate various risk metrics.

@author: Pedro Gronda Garrigues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from model import MarkowitzModel

class MonteCarloEnhancedMarkowitzModel(MarkowitzModel):
    """
    An enhanced version of the Markowitz model that uses Monte Carlo
    simulation techniques to analyze portfolio performance and risk.
    """
    
    def __init__(self, risk_free_rate=0.02):
        """
        Initialize the Monte Carlo-enhanced Markowitz model.
        
        Parameters:
        -----------
        risk_free_rate : float, optional
            The risk-free interest rate (annual), by default 0.02 (2%)
        """
        super().__init__(risk_free_rate)
        
    def simulate_gbm(self, num_simulations=1000, time_horizon=252, initial_value=1.0, seed=None):
        """
        Simulate portfolio value using Geometric Brownian Motion (GBM).
        
        Parameters:
        -----------
        num_simulations : int, optional
            Number of simulations to run, by default 1000
        time_horizon : int, optional
            Time horizon in days, by default 252 (1 year)
        initial_value : float, optional
            Initial portfolio value, by default 1.0
        seed : int, optional
            Random seed for reproducibility, by default None
            
        Returns:
        --------
        numpy.ndarray
            Array of simulated portfolio values
        """
        if self.optimal_portfolio is None:
            self.generate_efficient_frontier()
        
        if seed is not None:
            np.random.seed(seed)
        
        weights = np.array(list(self.optimal_portfolio['weights'].values()))
        
        mean_returns = self.returns.mean().values
        cov_matrix = self.cov_matrix.values / 252  # Daily covariance
        
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        
        simulation_results = np.zeros((num_simulations, time_horizon + 1))
        simulation_results[:, 0] = initial_value
        
        # Simulate using Geometric Brownian Motion
        # dS = mu * S * dt + sigma * S * dW
        dt = 1/252  # Daily time step (assuming 252 trading days in a year)
        
        for t in range(1, time_horizon + 1):
            Z = np.random.normal(0, 1, num_simulations)
            
            simulation_results[:, t] = simulation_results[:, t-1] * (
                1 + portfolio_return * dt + portfolio_volatility * np.sqrt(dt) * Z
            )
        
        return simulation_results
    
    def simulate_bootstrap(self, num_simulations=1000, time_horizon=252, initial_value=1.0, block_size=20, seed=None):
        """
        Simulate portfolio value using bootstrap resampling of historical returns.
        
        Parameters:
        -----------
        num_simulations : int, optional
            Number of simulations to run, by default 1000
        time_horizon : int, optional
            Time horizon in days, by default 252 (1 year)
        initial_value : float, optional
            Initial portfolio value, by default 1.0
        block_size : int, optional
            Size of blocks for block bootstrap, by default 20
        seed : int, optional
            Random seed for reproducibility, by default None
            
        Returns:
        --------
        numpy.ndarray
            Array of simulated portfolio values
        """
        if self.optimal_portfolio is None:
            self.generate_efficient_frontier()
        
        if seed is not None:
            np.random.seed(seed)
        
        weights = np.array(list(self.optimal_portfolio['weights'].values()))
        
        portfolio_returns = np.sum(self.returns.values * weights, axis=1)
        
        simulation_results = np.zeros((num_simulations, time_horizon + 1))
        simulation_results[:, 0] = initial_value
        
        num_returns = len(portfolio_returns)
        
        for i in range(num_simulations):
            portfolio_value = initial_value
            
            for t in range(1, time_horizon + 1):
                # Randomly sample from historical returns
                idx = np.random.randint(0, num_returns)
                return_t = portfolio_returns[idx]
                
                portfolio_value *= (1 + return_t)
                simulation_results[i, t] = portfolio_value
        
        return simulation_results
    
    def calculate_risk_metrics(self, simulation_results, initial_value=1.0, alpha=0.05):
        """
        Calculate various risk metrics from Monte Carlo simulation results.
        
        Parameters:
        -----------
        simulation_results : numpy.ndarray
            Array of simulated portfolio values
        initial_value : float, optional
            Initial portfolio value, by default 1.0
        alpha : float, optional
            Significance level for VaR and ES, by default 0.05 (5%)
            
        Returns:
        --------
        dict
            Dictionary containing various risk metrics
        """
        final_values = simulation_results[:, -1]
        
        returns = (final_values - initial_value) / initial_value
        
        metrics = {}
        
        metrics['mean'] = np.mean(returns)
        metrics['median'] = np.median(returns)
        metrics['std'] = np.std(returns)
        metrics['min'] = np.min(returns)
        metrics['max'] = np.max(returns)
        
        # Value at Risk (VaR)
        metrics['VaR'] = -np.percentile(returns, alpha * 100)
        
        # Expected Shortfall (ES) / Conditional VaR (CVaR)
        metrics['ES'] = -np.mean(returns[returns <= -metrics['VaR']])
        
        # Probability of loss
        metrics['prob_loss'] = np.mean(returns < 0)
        
        # Maximum drawdown
        max_drawdowns = np.zeros(simulation_results.shape[0])
        for i in range(simulation_results.shape[0]):
            running_max = np.maximum.accumulate(simulation_results[i, :])
            
            drawdown = (running_max - simulation_results[i, :]) / running_max
            
            max_drawdowns[i] = np.max(drawdown)
        
        metrics['max_drawdown'] = np.mean(max_drawdowns)
        
        # Sharpe Ratio
        metrics['sharpe_ratio'] = (metrics['mean'] - self.risk_free_rate) / metrics['std']
        
        return metrics
    
    def plot_simulation_results(self, simulation_results, percentiles=[5, 50, 95], figsize=(12, 8)):
        """
        Plot the results of a Monte Carlo simulation.
        
        Parameters:
        -----------
        simulation_results : numpy.ndarray
            Array of simulated portfolio values
        percentiles : list, optional
            Percentiles to plot, by default [5, 50, 95]
        figsize : tuple, optional
            Figure size, by default (12, 8)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        time_horizon = simulation_results.shape[1] - 1
        
        # Plot a subset of simulations for better visualization
        num_to_plot = min(100, simulation_results.shape[0])
        for i in range(num_to_plot):
            ax.plot(range(time_horizon + 1), simulation_results[i], 'b-', alpha=0.1)
        
        # Plot percentile lines
        for p in percentiles:
            percentile_values = np.percentile(simulation_results, p, axis=0)
            ax.plot(range(time_horizon + 1), percentile_values, 'g-', linewidth=2, label=f'{p}th Percentile')
        
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Portfolio Value')
        ax.set_title('Monte Carlo Simulation of Portfolio Performance')
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def plot_return_distribution(self, simulation_results, initial_value=1.0, figsize=(12, 8)):
        """
        Plot the distribution of portfolio returns from Monte Carlo simulation.
        
        Parameters:
        -----------
        simulation_results : numpy.ndarray
            Array of simulated portfolio values
        initial_value : float, optional
            Initial portfolio value, by default 1.0
        figsize : tuple, optional
            Figure size, by default (12, 8)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        
        final_values = simulation_results[:, -1]
        
        returns = (final_values - initial_value) / initial_value
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.histplot(returns, bins=50, kde=True, ax=ax)
        
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        var_5 = np.percentile(returns, 5)
        var_1 = np.percentile(returns, 1)
        
        ax.axvline(mean_return, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.4f}')
        ax.axvline(median_return, color='g', linestyle='--', linewidth=2, label=f'Median: {median_return:.4f}')
        ax.axvline(var_5, color='orange', linestyle='--', linewidth=2, label=f'VaR 5%: {-var_5:.4f}')
        ax.axvline(var_1, color='purple', linestyle='--', linewidth=2, label=f'VaR 1%: {-var_1:.4f}')
        
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Portfolio Returns')
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def plot_drawdown_distribution(self, simulation_results, figsize=(12, 8)):
        """
        Plot the distribution of portfolio drawdowns from Monte Carlo simulation.
        
        Parameters:
        -----------
        simulation_results : numpy.ndarray
            Array of simulated portfolio values
        figsize : tuple, optional
            Figure size, by default (12, 8)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        
        max_drawdowns = np.zeros(simulation_results.shape[0])
        
        for i in range(simulation_results.shape[0]):
            running_max = np.maximum.accumulate(simulation_results[i, :])
            
            drawdown = (running_max - simulation_results[i, :]) / running_max
            
            max_drawdowns[i] = np.max(drawdown)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.histplot(max_drawdowns, bins=50, kde=True, ax=ax)
        
        mean_drawdown = np.mean(max_drawdowns)
        median_drawdown = np.median(max_drawdowns)
        worst_drawdown = np.max(max_drawdowns)
        p95_drawdown = np.percentile(max_drawdowns, 95)
        
        ax.axvline(mean_drawdown, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_drawdown:.4f}')
        ax.axvline(median_drawdown, color='g', linestyle='--', linewidth=2, label=f'Median: {median_drawdown:.4f}')
        ax.axvline(worst_drawdown, color='purple', linestyle='--', linewidth=2, label=f'Worst: {worst_drawdown:.4f}')
        ax.axvline(p95_drawdown, color='orange', linestyle='--', linewidth=2, label=f'95th Percentile: {p95_drawdown:.4f}')
        
        ax.set_xlabel('Maximum Drawdown')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Maximum Drawdowns')
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def compare_simulation_methods(self, time_horizon=252, num_simulations=1000, initial_value=1.0, seed=42):
        """
        Compare different simulation methods.
        
        Parameters:
        -----------
        time_horizon : int, optional
            Time horizon in days, by default 252 (1 year)
        num_simulations : int, optional
            Number of simulations to run, by default 1000
        initial_value : float, optional
            Initial portfolio value, by default 1.0
        seed : int, optional
            Random seed for reproducibility, by default 42
            
        Returns:
        --------
        dict
            Dictionary containing the simulation results for each method
        """
        
        print("Running GBM simulation...")
        gbm_results = self.simulate_gbm(
            num_simulations=num_simulations,
            time_horizon=time_horizon,
            initial_value=initial_value,
            seed=seed
        )
        
        print("Running bootstrap simulation...")
        bootstrap_results = self.simulate_bootstrap(
            num_simulations=num_simulations,
            time_horizon=time_horizon,
            initial_value=initial_value,
            seed=seed
        )
        
        print("Calculating risk metrics...")
        gbm_metrics = self.calculate_risk_metrics(gbm_results)
        bootstrap_metrics = self.calculate_risk_metrics(bootstrap_results)
        
        # Combine results
        results = {
            'gbm': {
                'simulation': gbm_results,
                'metrics': gbm_metrics
            },
            'bootstrap': {
                'simulation': bootstrap_results,
                'metrics': bootstrap_metrics
            }
        }
        
        return results
    
    def plot_method_comparison(self, comparison_results, figsize=(18, 12)):
        """
        Plot a comparison of different simulation methods.
        
        Parameters:
        -----------
        comparison_results : dict
            Dictionary containing the simulation results for each method
        figsize : tuple, optional
            Figure size, by default (18, 12)
            
        Returns:
        --------
        tuple
            (matplotlib.figure.Figure, pandas.DataFrame) containing the figure and metrics dataframe
        """
        
        methods = list(comparison_results.keys())
        
        metrics_df = pd.DataFrame(index=methods)
        
        for method in methods:
            metrics = comparison_results[method]['metrics']
            metrics_df.loc[method, 'Mean Return'] = metrics['mean']
            metrics_df.loc[method, 'Std Dev'] = metrics['std']
            metrics_df.loc[method, 'VaR (5%)'] = metrics['VaR']
            metrics_df.loc[method, 'ES (5%)'] = metrics['ES']
            metrics_df.loc[method, 'Max Drawdown'] = metrics['max_drawdown']
            metrics_df.loc[method, 'Sharpe Ratio'] = metrics['sharpe_ratio']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        axes = axes.flatten()
        
        # Plot 1: Return distribution comparison
        ax = axes[0]
        
        for method in methods:
            final_values = comparison_results[method]['simulation'][:, -1]
            initial_value = comparison_results[method]['simulation'][0, 0]
            returns = (final_values - initial_value) / initial_value
            
            sns.kdeplot(returns, ax=ax, label=method.upper())
        
        ax.set_xlabel('Return')
        ax.set_ylabel('Density')
        ax.set_title('Return Distribution Comparison')
        ax.grid(True)
        ax.legend()
        
        # Plot 2: Risk metrics comparison
        ax = axes[1]
        
        x = np.arange(len(methods))
        width = 0.35
        
        var_values = [comparison_results[method]['metrics']['VaR'] for method in methods]
        es_values = [comparison_results[method]['metrics']['ES'] for method in methods]
        
        ax.bar(x - width/2, var_values, width, label='VaR (5%)')
        ax.bar(x + width/2, es_values, width, label='ES (5%)')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Value')
        ax.set_title('Risk Metrics Comparison - VaR and ES')
        ax.set_xticks(x)
        ax.set_xticklabels([method.upper() for method in methods])
        ax.grid(True)
        ax.legend()
        
        # Plot 3: Path comparison (average paths)
        ax = axes[2]
        
        time_horizon = comparison_results[methods[0]]['simulation'].shape[1] - 1
        
        for method in methods:
            avg_path = np.mean(comparison_results[method]['simulation'], axis=0)
            
            ax.plot(range(time_horizon + 1), avg_path, linewidth=2, label=method.upper())
        
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Portfolio Value')
        ax.set_title('Average Path Comparison')
        ax.grid(True)
        ax.legend()
        
        # Plot 4: Maximum drawdown comparison
        ax = axes[3]
        
        max_dd_values = [comparison_results[method]['metrics']['max_drawdown'] for method in methods]
        
        ax.bar(x, max_dd_values, width)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Maximum Drawdown')
        ax.set_title('Risk Metrics Comparison - Maximum Drawdown')
        ax.set_xticks(x)
        ax.set_xticklabels([method.upper() for method in methods])
        ax.grid(True)
        
        plt.tight_layout()
        
        return fig, metrics_df
