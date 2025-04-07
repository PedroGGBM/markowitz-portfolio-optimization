"""
Markowitz Portfolio Optimization - Main Script

This script serves as the main entry point for the Markowitz portfolio optimization project.
It provides a command-line interface to access the implemented features.

@author: Pedro Gronda Garrigues
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import the Markowitz model
from model import MarkowitzModel

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Markowitz Portfolio Optimization')
    
    # Data source options
    data_group = parser.add_argument_group('Data Source')
    data_source = data_group.add_mutually_exclusive_group(required=True)
    data_source.add_argument('--csv', action='store_true', help='Use CSV files as data source')
    data_source.add_argument('--sample', action='store_true', help='Use sample data for demonstration')
    
    # General options
    parser.add_argument('--risk-free-rate', type=float, default=0.02, help='Risk-free interest rate (default: 0.02)')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for plots and results (default: output)')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing CSV files (default: data)')
    
    args = parser.parse_args()
    
    return args

def create_model(args):
    """
    Create a model instance based on the specified arguments.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command-line arguments
        
    Returns:
    --------
    MarkowitzModel
        Model instance
    """
    model = MarkowitzModel(risk_free_rate=args.risk_free_rate)
    
    return model

def load_data(model, args):
    """
    Load data into the model based on the specified data source.
    
    Parameters:
    -----------
    model : MarkowitzModel
        Model instance
    args : argparse.Namespace
        Parsed command-line arguments
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the daily returns of each asset
    """
    if args.csv:
        print("Loading data from CSV files...")
        data_dir = os.path.abspath(args.data_dir)
        
        if not os.path.exists(data_dir):
            print(f"Data directory '{data_dir}' does not exist")
            sys.exit(1)
        
        if not any(f.endswith('.csv') for f in os.listdir(data_dir)):
            print(f"No CSV files found in '{data_dir}'")
            sys.exit(1)
        
        returns_df = model.load_data_from_csv(data_dir)
    
    elif args.sample:
        print("Creating sample data for demonstration...")
        
        np.random.seed(42)
        num_assets = 5
        num_days = 1000
        
        returns = np.random.normal(0.001, 0.02, (num_days, num_assets))
        assets = ['Asset_' + str(i+1) for i in range(num_assets)]
        
        returns_df = pd.DataFrame(returns, columns=assets)
        
        model.returns = returns_df
        model.assets = assets
        model.cov_matrix = returns_df.cov() * 252
    
    return returns_df

def run_analysis(model, output_dir):
    """
    Run analysis using the Markowitz model.
    
    Parameters:
    -----------
    model : MarkowitzModel
        Model instance
    output_dir : str
        Output directory for plots and results
        
    Returns:
    --------
    dict
        Dictionary containing the results
    """
    print("Generating the efficient frontier...")
    efficient_frontier = model.generate_efficient_frontier(num_portfolios=100)
    
    optimal_portfolio = model.get_optimal_portfolio()
    
    print("\nOptimal Portfolio:")
    print(f"Expected Return: {optimal_portfolio['return']:.4f}")
    print(f"Volatility: {optimal_portfolio['volatility']:.4f}")
    print(f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.4f}")
    print("Weights:")
    for asset, weight in optimal_portfolio['weights'].items():
        print(f"  {asset}: {weight:.4f}")
    
    print("Plotting the efficient frontier...")
    fig = model.plot_efficient_frontier(show_assets=True, show_optimal=True)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'efficient_frontier.png')
    fig.savefig(output_file)
    print(f"Efficient frontier plot saved to {output_file}")
    
    return {
        'efficient_frontier': efficient_frontier,
        'optimal_portfolio': optimal_portfolio,
        'figure': fig
    }

def main():
    """
    Main function to run the Markowitz portfolio optimization.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model instance
    model = create_model(args)
    
    # Load data
    returns_df = load_data(model, args)
    
    # Print basic statistics
    print("\nBasic Statistics:")
    print(f"Number of assets: {len(model.assets)}")
    print(f"Assets: {', '.join(model.assets)}")
    if hasattr(returns_df, 'index') and hasattr(returns_df.index, '__getitem__'):
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
    
    # Run analysis
    results = run_analysis(model, args.output_dir)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
