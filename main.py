"""
Markowitz Portfolio Optimization - Main Script

This script serves as the main entry point for the Markowitz portfolio optimization project.
It provides a command-line interface to access all the implemented features and allows
the user to choose which enhancement(s) to use.

@author: Pedro Gronda Garrigues
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import all models
from model import MarkowitzModel
from ml_enhancement import MLEnhancedMarkowitzModel
from factor_modeling import FactorEnhancedMarkowitzModel
from monte_carlo_enhancement import MonteCarloEnhancedMarkowitzModel
from api_integration import APIEnhancedMarkowitzModel

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """

    parser = argparse.ArgumentParser(description='Markowitz Portfolio Optimization')
    
    # [data source] options
    data_group = parser.add_argument_group('Data Source')
    data_source = data_group.add_mutually_exclusive_group(required=True)
    data_source.add_argument('--csv', action='store_true', help='Use CSV files as data source')
    data_source.add_argument('--api', action='store_true', help='Use Alpha Vantage API as data source')
    data_source.add_argument('--sample', action='store_true', help='Use sample data for demonstration')
    
    # [enhancement] options
    enhancement_group = parser.add_argument_group('Enhancements')
    enhancement_group.add_argument('--ml', action='store_true', help='Use machine learning enhancement')
    enhancement_group.add_argument('--factor', action='store_true', help='Use factor modeling enhancement')
    enhancement_group.add_argument('--monte-carlo', action='store_true', help='Use Monte Carlo enhancement')
    enhancement_group.add_argument('--all', action='store_true', help='Use all enhancements')
    
    # [general] options
    parser.add_argument('--risk-free-rate', type=float, default=0.02, help='Risk-free interest rate (default: 0.02)')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for plots and results (default: output)')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing CSV files (default: data)')
    parser.add_argument('--symbols', type=str, nargs='+', help='Stock symbols to use (required for API)')
    parser.add_argument('--api-key', type=str, help='Alpha Vantage API key (required for API)')
    parser.add_argument('--config-file', type=str, default='config/api_keys.json', help='Config file containing API keys (default: config/api_keys.json)')
    
    args = parser.parse_args()
    
    # args sanity check
    if args.api and not (args.api_key or os.path.exists(args.config_file)):
        parser.error('--api requires --api-key or a valid config file')
    
    if args.api and not args.symbols:
        parser.error('--api requires --symbols')
    
    return args

def load_api_key(config_file):
    """
    Load Alpha Vantage API key from a JSON file.
    
    Parameters:
    -----------
    config_file : str
        Path to the JSON file containing the API key
        
    Returns:
    --------
    str
        Alpha Vantage API key
    """

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if 'alpha_vantage_api_key' in config:
            return config['alpha_vantage_api_key']
        else:
            print("API key not found in config file")
            return None
    
    except Exception as e:
        print(f"Error loading API key: {e}")
        return None

def create_model(args):
    """
    Create a model instance based on the specified enhancements.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command-line arguments
        
    Returns:
    --------
    MarkowitzModel
        Model instance
    """

    if args.all:
        # ff all enhancements are requested ->use a combined model
        # default to API-enhance model (for simplicity)
        if args.api:
            api_key = args.api_key or load_api_key(args.config_file)
            model = APIEnhancedMarkowitzModel(risk_free_rate=args.risk_free_rate, api_key=api_key)
        else:
            model = APIEnhancedMarkowitzModel(risk_free_rate=args.risk_free_rate)
    elif args.ml:
        model = MLEnhancedMarkowitzModel(risk_free_rate=args.risk_free_rate)
    elif args.factor:
        model = FactorEnhancedMarkowitzModel(risk_free_rate=args.risk_free_rate)
    elif args.monte_carlo:
        model = MonteCarloEnhancedMarkowitzModel(risk_free_rate=args.risk_free_rate)
    elif args.api:
        api_key = args.api_key or load_api_key(args.config_file)
        model = APIEnhancedMarkowitzModel(risk_free_rate=args.risk_free_rate, api_key=api_key)
    else:
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
    
    elif args.api:
        print("Loading data from Alpha Vantage API...")
        
        if not isinstance(model, APIEnhancedMarkowitzModel):
            print("API data source requires API-enhanced model")
            sys.exit(1)
        
        returns_df = model.update_model_with_api_data(args.symbols, data_type='daily', output_size='compact')
    
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

def run_basic_analysis(model, output_dir):
    """
    Run basic analysis using the Markowitz model.
    
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
    fig = model.plot_efficient_frontier(show_assets=True, show_cal=True, show_optimal=True, random_portfolios=500)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'efficient_frontier.png')
    fig.savefig(output_file)
    print(f"Efficient frontier plot saved to {output_file}")
    
    return {
        'efficient_frontier': efficient_frontier,
        'optimal_portfolio': optimal_portfolio,
        'figure': fig
    }

def run_ml_analysis(model, output_dir):
    """
    Run analysis using the ML-enhanced Markowitz model.
    
    Parameters:
    -----------
    model : MLEnhancedMarkowitzModel
        Model instance
    output_dir : str
        Output directory for plots and results
        
    Returns:
    --------
    dict
        Dictionary containing the results
    """

    if not isinstance(model, MLEnhancedMarkowitzModel):
        print("STATUS: ML analysis requires ML-enhanced model")
        return None
    
    # ARIMA Return
    print("Predicting returns using ARIMA...")
    predicted_returns = model.predict_returns_arima()
    
    # GARCH Volatility
    print("Predicting volatility using GARCH...")
    predicted_cov_matrix = model.predict_volatility_garch()
    
    # historic value comparison of predictions
    print("Comparing predictions with historical values...")
    returns_comparison, volatility_comparison = model.compare_predictions_with_historical()
    
    print("\nReturns Comparison:")
    print(returns_comparison)
    
    print("\nVolatility Comparison:")
    print(volatility_comparison)
    
    print("Plotting comparison...")
    fig_comparison = model.plot_comparison()
    output_file_comparison = os.path.join(output_dir, 'ml_comparison.png')
    fig_comparison.savefig(output_file_comparison)
    print(f"Comparison plot saved to {output_file_comparison}")
    
    print("Generating efficient frontier with predictions...")
    predicted_frontier = model.generate_efficient_frontier_with_predictions()
    
    print("Plotting efficient frontiers comparison...")
    fig_frontiers = model.plot_efficient_frontiers_comparison()
    output_file_frontiers = os.path.join(output_dir, 'ml_efficient_frontiers.png')
    fig_frontiers.savefig(output_file_frontiers)
    print(f"Efficient frontiers comparison plot saved to {output_file_frontiers}")
    
    return {
        'predicted_returns': model.predicted_returns,
        'predicted_cov_matrix': model.predicted_cov_matrix,
        'returns_comparison': returns_comparison,
        'volatility_comparison': volatility_comparison,
        'predicted_frontier': predicted_frontier,
        'figures': {
            'comparison': fig_comparison,
            'frontiers': fig_frontiers
        }
    }

def run_factor_analysis(model, output_dir):
    """
    Run analysis using the factor-enhanced Markowitz model.
    
    Parameters:
    -----------
    model : FactorEnhancedMarkowitzModel
        Model instance
    output_dir : str
        Output directory for plots and results
        
    Returns:
    --------
    dict
        Dictionary containing the results
    """
    
    if not isinstance(model, FactorEnhancedMarkowitzModel):
        print("Factor analysis requires factor-enhanced model")
        return None
    
    print("Generating synthetic factor data...")
    factor_data = model.generate_synthetic_factor_data()
    
    print("Estimating factor loadings...")
    factor_loadings = model.estimate_factor_loadings()
    
    print("\nFactor Loadings:")
    print(factor_loadings)
    
    print("\nEstimating factor expected returns...")
    factor_expected_returns = model.estimate_factor_expected_returns()
    
    print("\nFactor Expected Returns:")
    print(factor_expected_returns)
    
    print("\nEstimating factor covariance matrix...")
    factor_cov_matrix = model.estimate_factor_cov_matrix()
    
    print("\nFactor Covariance Matrix:")
    print(factor_cov_matrix)
    
    print("\nAnalyzing factor contributions...")
    contributions = model.analyze_factor_contributions()
    
    print("\nFactor Contributions:")
    print(contributions)
    
    print("\nPlotting factor loadings...")
    fig_loadings = model.plot_factor_loadings()
    output_file_loadings = os.path.join(output_dir, 'factor_loadings.png')
    fig_loadings.savefig(output_file_loadings)
    print(f"Factor loadings plot saved to {output_file_loadings}")
    
    print("\nPlotting factor contributions...")
    fig_contributions = model.plot_factor_contributions()
    output_file_contributions = os.path.join(output_dir, 'factor_contributions.png')
    fig_contributions.savefig(output_file_contributions)
    print(f"Factor contributions plot saved to {output_file_contributions}")
    
    print("\nGenerating efficient frontier with factors...")
    factor_frontier = model.generate_efficient_frontier_with_factors()
    
    print("\nPlotting efficient frontiers comparison...")
    fig_frontiers = model.plot_efficient_frontiers_comparison()
    output_file_frontiers = os.path.join(output_dir, 'factor_efficient_frontiers.png')
    fig_frontiers.savefig(output_file_frontiers)
    print(f"Efficient frontiers comparison plot saved to {output_file_frontiers}")
    
    return {
        'factor_data': factor_data,
        'factor_loadings': factor_loadings,
        'factor_expected_returns': factor_expected_returns,
        'factor_cov_matrix': factor_cov_matrix,
        'contributions': contributions,
        'factor_frontier': factor_frontier,
        'figures': {
            'loadings': fig_loadings,
            'contributions': fig_contributions,
            'frontiers': fig_frontiers
        }
    }

def run_monte_carlo_analysis(model, output_dir):
    """
    Run analysis using the Monte Carlo-enhanced Markowitz model.
    
    Parameters:
    -----------
    model : MonteCarloEnhancedMarkowitzModel
        Model instance
    output_dir : str
        Output directory for plots and results
        
    Returns:
    --------
    dict
        Dictionary containing the results
    """
    
    if not isinstance(model, MonteCarloEnhancedMarkowitzModel):
        print("Monte Carlo analysis requires Monte Carlo-enhanced model")
        return None
    
    print("Generating the efficient frontier...")
    efficient_frontier = model.generate_efficient_frontier()
    
    print("Running Monte Carlo simulations using different methods...")
    comparison_results = model.compare_simulation_methods(
        time_horizon=252,  # 1 year
        num_simulations=1000,
        initial_value=1.0,
        seed=42
    )
    
    print("Calculating risk metrics...")
    gbm_metrics = comparison_results['gbm']['metrics']
    
    print("\nGBM Risk Metrics:")
    print(f"Mean Return: {gbm_metrics['mean']:.4f}")
    print(f"Volatility: {gbm_metrics['std']:.4f}")
    print(f"VaR (5%): {gbm_metrics['VaR']:.4f}")
    print(f"ES (5%): {gbm_metrics['ES']:.4f}")
    print(f"Max Drawdown: {gbm_metrics['max_drawdown']:.4f}")
    print(f"Sharpe Ratio: {gbm_metrics['sharpe_ratio']:.4f}")
    
    print("\nPlotting simulation results...")
    fig_simulation = model.plot_simulation_results(comparison_results['gbm']['simulation'])
    output_file_simulation = os.path.join(output_dir, 'monte_carlo_simulation.png')
    fig_simulation.savefig(output_file_simulation)
    print(f"Simulation results plot saved to {output_file_simulation}")
    
    print("\nPlotting return distribution...")
    fig_return_dist = model.plot_return_distribution(comparison_results['gbm']['simulation'])
    output_file_return_dist = os.path.join(output_dir, 'mc_gbm_returns.png')
    fig_return_dist.savefig(output_file_return_dist)
    print(f"Return distribution plot saved to {output_file_return_dist}")
    
    print("\nPlotting drawdown distribution...")
    fig_drawdown_dist = model.plot_drawdown_distribution(comparison_results['gbm']['simulation'])
    output_file_drawdown_dist = os.path.join(output_dir, 'mc_gbm_drawdowns.png')
    fig_drawdown_dist.savefig(output_file_drawdown_dist)
    print(f"Drawdown distribution plot saved to {output_file_drawdown_dist}")
    
    print("\nPlotting method comparison...")
    fig_method_comp, metrics_df = model.plot_method_comparison(comparison_results)
    output_file_method_comp = os.path.join(output_dir, 'mc_method_comparison.png')
    fig_method_comp.savefig(output_file_method_comp)
    print(f"Method comparison plot saved to {output_file_method_comp}")
    
    print("\nComparison Metrics:")
    print(metrics_df)
    
    return {
        'comparison_results': comparison_results,
        'metrics_df': metrics_df,
        'figures': {
            'simulation': fig_simulation,
            'return_dist': fig_return_dist,
            'drawdown_dist': fig_drawdown_dist,
            'method_comp': fig_method_comp
        }
    }

def run_api_analysis(model, output_dir, symbols):
    """
    Run analysis using the API-enhanced Markowitz model.
    
    Parameters:
    -----------
    model : APIEnhancedMarkowitzModel
        Model instance
    output_dir : str
        Output directory for plots and results
    symbols : list
        List of stock symbols
        
    Returns:
    --------
    dict
        Dictionary containing the results
    """
    
    if not isinstance(model, APIEnhancedMarkowitzModel):
        print("API analysis requires API-enhanced model")
        return None
    
    print("Plotting real-time efficient frontier...")
    fig_frontier = model.plot_real_time_efficient_frontier()
    output_file_frontier = os.path.join(output_dir, 'api_efficient_frontier.png')
    fig_frontier.savefig(output_file_frontier)
    print(f"Real-time efficient frontier plot saved to {output_file_frontier}")
    
    print("Plotting price history...")
    fig_prices = model.plot_price_history()
    output_file_prices = os.path.join(output_dir, 'api_price_history.png')
    fig_prices.savefig(output_file_prices)
    print(f"Price history plot saved to {output_file_prices}")
    
    print("Plotting correlation matrix...")
    fig_corr = model.plot_correlation_matrix()
    output_file_corr = os.path.join(output_dir, 'api_correlation_matrix.png')
    fig_corr.savefig(output_file_corr)
    print(f"Correlation matrix plot saved to {output_file_corr}")
    
    return {
        'figures': {
            'frontier': fig_frontier,
            'prices': fig_prices,
            'corr': fig_corr
        }
    }

def main():
    """
    Main function to run the Markowitz portfolio optimization.
    """
    
    # parse: command-line arguments
    args = parse_arguments()
    
    # create: output directory 
    os.makedirs(args.output_dir, exist_ok=True)
    
    # create: model instance
    model = create_model(args)
    
    # load: data
    returns_df = load_data(model, args)
    
    # run: basic analysis
    basic_results = run_basic_analysis(model, args.output_dir)
    
    # run: enhancement-specific analyses
    if args.ml or args.all:
        print("\n--- Running Machine Learning Enhancement ---")
        ml_results = run_ml_analysis(model, args.output_dir)
    
    if args.factor or args.all:
        print("\n--- Running Factor Modeling Enhancement ---")
        factor_results = run_factor_analysis(model, args.output_dir)
    
    if args.monte_carlo or args.all:
        print("\n--- Running Monte Carlo Enhancement ---")
        mc_results = run_monte_carlo_analysis(model, args.output_dir)
    
    if args.api:
        print("\n--- Running API Integration ---")
        api_results = run_api_analysis(model, args.output_dir, args.symbols)
    
    print("\nAll analyses completed successfully!")

if __name__ == "__main__":
    main()