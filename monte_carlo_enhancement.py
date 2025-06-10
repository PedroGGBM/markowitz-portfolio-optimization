"""
Advanced Monte Carlo Simulation Enhancement

This script demonstrates how to enhance the Markowitz model with advanced Monte Carlo
simulation techniques to analyze the robustness of the portfolio across different
market scenarios and calculate various risk metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from model import MarkowitzModel

class MonteCarloEnhancedMarkowitzModel(MarkowitzModel):
    """
    An enhanced version of the Markowitz model that uses advanced Monte Carlo
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
            raise ValueError("Optimal portfolio must be calculated before running Monte Carlo simulation")
        
        if seed is not None:
            np.random.seed(seed)
        
        weights = np.array(list(self.optimal_portfolio['weights'].values()))
        
        mean_returns = self.returns.mean().values
        cov_matrix = self.cov_matrix.values
        
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        simulation_results = np.zeros((num_simulations, time_horizon + 1))
        simulation_results[:, 0] = initial_value
        
        # Simulate using Geometric Brownian Motion
        # dS = mu * S * dt + sigma * S * dW
        #### S is the portfolio value, 
        #### mu is the drift,
        #### sigma is the volatility,
        #### dt is the time step,
        #### dW is the Wiener process increment
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
            raise ValueError("Optimal portfolio must be calculated before running Monte Carlo simulation")
        
        if seed is not None:
            np.random.seed(seed)
        
        weights = np.array(list(self.optimal_portfolio['weights'].values()))
        
        portfolio_returns = np.sum(self.returns.values * weights, axis=1)
        
        simulation_results = np.zeros((num_simulations, time_horizon + 1))
        simulation_results[:, 0] = initial_value
        
        num_returns = len(portfolio_returns)
        
        num_blocks = int(np.ceil(time_horizon / block_size))
        
        for i in range(num_simulations):
            portfolio_value = initial_value
            
            for j in range(num_blocks):
                start_idx = np.random.randint(0, num_returns - block_size + 1)
                
                block_returns = portfolio_returns[start_idx:start_idx + block_size]
                
                days_to_use = min(block_size, time_horizon - j * block_size)
                
                for k in range(days_to_use):
                    t = j * block_size + k + 1
                    if t <= time_horizon:
                        portfolio_value *= (1 + block_returns[k])
                        simulation_results[i, t] = portfolio_value
        
        return simulation_results
    
    def simulate_multivariate(self, num_simulations=1000, time_horizon=252, initial_value=1.0, method='normal', seed=None):
        """
        Simulate portfolio value using multivariate simulation of asset returns.
        
        Parameters:
        -----------
        num_simulations : int, optional
            Number of simulations to run, by default 1000
        time_horizon : int, optional
            Time horizon in days, by default 252 (1 year)
        initial_value : float, optional
            Initial portfolio value, by default 1.0
        method : str, optional
            Method for generating random returns, by default 'normal'
            Options: 'normal', 't', 'copula'
        seed : int, optional
            Random seed for reproducibility, by default None
            
        Returns:
        --------
        numpy.ndarray
            Array of simulated portfolio values
        """
        if self.optimal_portfolio is None:
            raise ValueError("Optimal portfolio must be calculated before running Monte Carlo simulation")
        
        if seed is not None:
            np.random.seed(seed)
        
        weights = np.array(list(self.optimal_portfolio['weights'].values()))
        
        mean_returns = self.returns.mean().values
        cov_matrix = self.cov_matrix.values / 252  # Daily covariance
        
        simulation_results = np.zeros((num_simulations, time_horizon + 1))
        simulation_results[:, 0] = initial_value
        
        if method == 'normal':
            random_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, (num_simulations, time_horizon)
            )
        elif method == 't':
            # random returns using multivariate t distribution
            # with 5 degrees of freedom (heavier tails than normal)
            df = 5 # DoF
            
            # multivariate normal samples
            normal_samples = np.random.multivariate_normal(
                np.zeros_like(mean_returns), cov_matrix, (num_simulations, time_horizon)
            )
            
            # chi-squared samples
            chi2_samples = np.random.chisquare(df, (num_simulations, time_horizon, 1))
            
            # convert to multivariate t samples
            t_samples = normal_samples / np.sqrt(chi2_samples / df)
            
            random_returns = t_samples + mean_returns
        elif method == 'copula':
            # generate random returns w/ Gaussian copula
            # (preserves the marginal distributions of the returns)
            
            # empirical CDF
            ecdf = {}
            for i in range(len(self.assets)):
                asset_returns = self.returns.iloc[:, i].values
                ecdf[i] = stats.ecdf(asset_returns)
            
            normal_samples = np.random.multivariate_normal(
                np.zeros(len(self.assets)), self.returns.corr().values, (num_simulations, time_horizon)
            )
            uniform_samples = stats.norm.cdf(normal_samples)
            
            random_returns = np.zeros_like(uniform_samples)
            for i in range(len(self.assets)):
                for j in range(num_simulations):
                    for k in range(time_horizon):
                        u = uniform_samples[j, k, i]
                        idx = np.searchsorted(ecdf[i][0], u)
                        if idx == 0:
                            random_returns[j, k, i] = self.returns.iloc[0, i]
                        else:
                            random_returns[j, k, i] = self.returns.iloc[idx-1, i]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        for t in range(1, time_horizon + 1):
            portfolio_returns = np.sum(random_returns[:, t-1, :] * weights, axis=1)
            
            simulation_results[:, t] = simulation_results[:, t-1] * (1 + portfolio_returns)
        
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
        
        for p in [1, 5, 10, 25, 75, 90, 95, 99]:
            metrics[f'percentile_{p}'] = np.percentile(returns, p)
        
        # Value at Risk (VaR)
        metrics['VaR'] = -np.percentile(returns, alpha * 100)
        
        # Expected Shortfall (ES) / Conditional VaR (CVaR)
        metrics['ES'] = -np.mean(returns[returns <= -metrics['VaR']])
        
        # probability of loss
        metrics['prob_loss'] = np.mean(returns < 0)
        
        max_drawdowns = np.zeros(simulation_results.shape[0])
        for i in range(simulation_results.shape[0]):
            running_max = np.maximum.accumulate(simulation_results[i, :])
            
            drawdown = (running_max - simulation_results[i, :]) / running_max
            
            max_drawdowns[i] = np.max(drawdown)
        
        metrics['max_drawdown'] = np.mean(max_drawdowns)
        
        # Sharpe Ratio
        metrics['sharpe_ratio'] = (metrics['mean'] - self.risk_free_rate) / metrics['std']
        
        # Sortino Ratio (using downside deviation)
        downside_returns = returns[returns < self.risk_free_rate]
        downside_deviation = np.sqrt(np.mean((downside_returns - self.risk_free_rate) ** 2))
        metrics['sortino_ratio'] = (metrics['mean'] - self.risk_free_rate) / downside_deviation if len(downside_returns) > 0 else np.nan
        
        # Omega Ratio
        threshold = self.risk_free_rate
        gains = returns[returns >= threshold] - threshold
        losses = threshold - returns[returns < threshold]
        metrics['omega_ratio'] = np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else np.inf
        
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
        
        num_to_plot = min(100, simulation_results.shape[0])
        for i in range(num_to_plot):
            ax.plot(range(time_horizon + 1), simulation_results[i], 'b-', alpha=0.1)
        
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
        
        print("Running multivariate normal simulation...")
        normal_results = self.simulate_multivariate(
            num_simulations=num_simulations,
            time_horizon=time_horizon,
            initial_value=initial_value,
            method='normal',
            seed=seed
        )
        
        print("Running multivariate t simulation...")
        t_results = self.simulate_multivariate(
            num_simulations=num_simulations,
            time_horizon=time_horizon,
            initial_value=initial_value,
            method='t',
            seed=seed
        )
        
        print("Calculating risk metrics...")
        gbm_metrics = self.calculate_risk_metrics(gbm_results)
        bootstrap_metrics = self.calculate_risk_metrics(bootstrap_results)
        normal_metrics = self.calculate_risk_metrics(normal_results)
        t_metrics = self.calculate_risk_metrics(t_results)
        
        # Combine results
        results = {
            'gbm': {
                'simulation': gbm_results,
                'metrics': gbm_metrics
            },
            'bootstrap': {
                'simulation': bootstrap_results,
                'metrics': bootstrap_metrics
            },
            'normal': {
                'simulation': normal_results,
                'metrics': normal_metrics
            },
            't': {
                'simulation': t_results,
                'metrics': t_metrics
            }
        }
        
        return results
    
    def plot_method_comparison(self, comparison_results, figsize=(18, 15)):
        """
        Plot a comparison of different simulation methods.
        
        Parameters:
        -----------
        comparison_results : dict
            Dictionary containing the simulation results for each method
        figsize : tuple, optional
            Figure size, by default (18, 15)
            
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
            metrics_df.loc[method, 'Sortino Ratio'] = metrics['sortino_ratio']
            metrics_df.loc[method, 'Omega Ratio'] = metrics['omega_ratio']
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        axes = axes.flatten()
        
        # [Plot 1] Percentile comparison across methods
        ax = axes[0]
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_data = np.zeros((len(methods), len(percentiles)))
        
        for i, method in enumerate(methods):
            for j, p in enumerate(percentiles):
                key = f'percentile_{p}'
                if key in comparison_results[method]['metrics']:
                    percentile_data[i, j] = comparison_results[method]['metrics'][key]
                else:
                    final_values = comparison_results[method]['simulation'][:, -1]
                    initial_value = comparison_results[method]['simulation'][0, 0]
                    returns = (final_values - initial_value) / initial_value
                    percentile_data[i, j] = np.percentile(returns, p)
        
        for i, method in enumerate(methods):
            ax.plot(percentiles, percentile_data[i], marker='o', linewidth=2, label=method.capitalize())
        
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Return')
        ax.set_title('Return Percentiles Across Methods')
        ax.grid(True)
        ax.legend()
        
        # [Plot 2] Return distribution comparison
        ax = axes[1]
        
        for method in methods:
            final_values = comparison_results[method]['simulation'][:, -1]
            initial_value = comparison_results[method]['simulation'][0, 0]
            returns = (final_values - initial_value) / initial_value
            
            sns.kdeplot(returns, ax=ax, label=method.capitalize())
        
        ax.set_xlabel('Return')
        ax.set_ylabel('Density')
        ax.set_title('Return Distribution Comparison')
        ax.grid(True)
        ax.legend()
        
        # [Plot 3] Return and volatility comparison
        ax = axes[2]
        
        x = [comparison_results[method]['metrics']['std'] for method in methods]
        y = [comparison_results[method]['metrics']['mean'] for method in methods]
        
        ax.scatter(x, y, s=100)
        
        for i, method in enumerate(methods):
            ax.annotate(method.capitalize(), (x[i], y[i]), xytext=(10, 0), 
                       textcoords='offset points', fontsize=12)
        
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Expected Return')
        ax.set_title('Return and Volatility Comparison')
        ax.grid(True)
        
        # [Plot 4] Risk metrics comparison - VaR and ES
        ax = axes[3]
        
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
        ax.set_xticklabels([method.capitalize() for method in methods])
        ax.grid(True)
        ax.legend()
        
        # [Plot 5] Risk metrics comparison - Max Drawdown
        ax = axes[4]
        
        max_dd_values = [comparison_results[method]['metrics']['max_drawdown'] for method in methods]
        
        ax.bar(x, max_dd_values, width)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Maximum Drawdown')
        ax.set_title('Risk Metrics Comparison - Maximum Drawdown')
        ax.set_xticks(x)
        ax.set_xticklabels([method.capitalize() for method in methods])
        ax.grid(True)
        
        # [Plot 6] Path comparison (average paths)
        ax = axes[5]
        
        time_horizon = comparison_results[methods[0]]['simulation'].shape[1] - 1
        
        for method in methods:
            avg_path = np.mean(comparison_results[method]['simulation'], axis=0)
            
            ax.plot(range(time_horizon + 1), avg_path, linewidth=2, label=method.capitalize())
        
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Portfolio Value')
        ax.set_title('Average Path Comparison')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        
        return fig, metrics_df