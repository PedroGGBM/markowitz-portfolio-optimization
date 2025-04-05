"""
Example Script

This script demonstrates how to use the Markowitz model with real data.
It loads data from CSV files, generates the efficient frontier, finds the optimal portfolio,
and plots the results.

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

    # Create Markowitz model instance
    model = MarkowitzModel(risk_free_rate=0.02)
    
    # Path to data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    # Check if data exists
    if not os.path.exists(data_dir) or not any(f.endswith('.csv') for f in os.listdir(data_dir)):
        print("No data found. Please run download_data.py first to download sample data.")
        print("You can run it with: python src_new/download_data.py")
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
    fig = model.plot_efficient_frontier(show_assets=True, show_optimal=True)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'efficient_frontier.png')
    fig.savefig(output_file)
    print(f"Efficient frontier plot saved to {output_file}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
