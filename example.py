"""
Example Script

This script demonstrates how to use the Markowitz model with real data.
It loads data from CSV files, generates the efficient frontier, finds the optimal portfolio,
plots the results, and runs a Monte Carlo simulation.

@author: Pedro Gronda Garrigues
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from model import MarkowitzModel

def main():
    """
    Main function to demonstrate the usage of the Markowitz model.
    """

    # create Markowitz model instance
    model = MarkowitzModel(risk_free_rate=0.02)
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    if not os.path.exists(data_dir) or not any(f.endswith('.csv') for f in os.listdir(data_dir)):
        print("STATUS: No data found. Please run download_data.py first to download sample data.")
        print("You can run it with: python src/download_data.py")
        return
    
    print("Loading data from CSV files...")
    returns_df = model.load_data_from_csv(data_dir)
    
    print("\nBasic Statistics:")
    print(f"Number of assets: {len(model.assets)}")
    print(f"Assets: {', '.join(model.assets)}")
    print(f"Date range: {returns_df.index[0]} to {returns_df.index[-1]}")
    print(f"Number of trading days: {len(returns_df)}")
    
    print("\nAnnualized Returns and Volatilities:")
    annual_returns = returns_df.mean() * 252
    annual_volatilities = returns_df.std() * (252 ** 0.5)
    
    for asset in model.assets:
        print(f"{asset}:")
        print(f"  Return: {annual_returns[asset]:.4f}")
        print(f"  Volatility: {annual_volatilities[asset]:.4f}")
        print(f"  Sharpe Ratio: {(annual_returns[asset] - model.risk_free_rate) / annual_volatilities[asset]:.4f}")
    
    print("\nGenerating the efficient frontier...")
    efficient_frontier = model.generate_efficient_frontier(num_portfolios=100)
    
    optimal_portfolio = model.get_optimal_portfolio()
    
    print("\nOptimal Portfolio:")
    print(f"Expected Return: {optimal_portfolio['return']:.4f}")
    print(f"Volatility: {optimal_portfolio['volatility']:.4f}")
    print(f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.4f}")
    print("Weights:")
    for asset, weight in optimal_portfolio['weights'].items():
        print(f"  {asset}: {weight:.4f}")
    
    # Plot the efficient frontier
    print("\nPlotting the efficient frontier...")
    fig = model.plot_efficient_frontier(show_assets=True, show_cal=True, show_optimal=True, random_portfolios=500)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'efficient_frontier.png')
    fig.savefig(output_file)
    print(f"Efficient frontier plot saved to {output_file}")
    
    print("\nRunning a Monte Carlo simulation...")
    from monte_carlo_enhancement import MonteCarloEnhancedMarkowitzModel
    
    mc_model = MonteCarloEnhancedMarkowitzModel(risk_free_rate=0.02)
    mc_model.returns = model.returns
    mc_model.assets = model.assets
    mc_model.cov_matrix = model.cov_matrix
    
    mc_model.generate_efficient_frontier()
    
    print("Running GBM simulation...")
    simulation_results = mc_model.simulate_gbm(
        num_simulations=1000,
        time_horizon=252,  # 1 year
        initial_value=1.0,
        seed=42
    )
    
    metrics = mc_model.calculate_risk_metrics(simulation_results)
    
    print("\nRisk Metrics:")
    print(f"Mean Return: {metrics['mean']:.4f}")
    print(f"Volatility: {metrics['std']:.4f}")
    print(f"VaR (5%): {metrics['VaR']:.4f}")
    print(f"ES (5%): {metrics['ES']:.4f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    
    print("\nPlotting simulation results...")
    fig_simulation = mc_model.plot_simulation_results(simulation_results)
    output_file_simulation = os.path.join(output_dir, 'mc_gbm_simulation.png')
    fig_simulation.savefig(output_file_simulation)
    print(f"Simulation results plot saved to {output_file_simulation}")
    
    print("\nPlotting return distribution...")
    fig_return_dist = mc_model.plot_return_distribution(simulation_results)
    output_file_return_dist = os.path.join(output_dir, 'mc_gbm_returns.png')
    fig_return_dist.savefig(output_file_return_dist)
    print(f"Return distribution plot saved to {output_file_return_dist}")
    
    print("\nPlotting drawdown distribution...")
    fig_drawdown_dist = mc_model.plot_drawdown_distribution(simulation_results)
    output_file_drawdown_dist = os.path.join(output_dir, 'mc_gbm_drawdowns.png')
    fig_drawdown_dist.savefig(output_file_drawdown_dist)
    print(f"Drawdown distribution plot saved to {output_file_drawdown_dist}")
    
    print("\nExample completed successfully!")