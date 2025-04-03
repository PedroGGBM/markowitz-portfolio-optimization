"""
Download Sample Data

This script downloads historical stock data for a few popular stocks using the yfinance library
and saves the data as CSV files in the data/ directory.

@author: Pedro Gronda Garrigues
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def download_stock_data(symbols, start_date, end_date, output_dir):
    """
    Download historical stock data for the given symbols and save as CSV files.
    
    Parameters:
    -----------
    symbols : list
        List of stock symbols to download data for
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    output_dir : str
        Directory to save the CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol in symbols:
        print(f"Downloading data for {symbol}...")
        
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"No data available for {symbol}")
                continue
            
            data = data.reset_index()
            
            output_file = os.path.join(output_dir, f"{symbol}.csv")
            data.to_csv(output_file, index=False)
            
            print(f"Data for {symbol} saved to {output_file}")
        
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")

def main():
    """
    Main function to download sample data.
    """
    
    # Define stock symbols to download
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']
    
    # Date range (5 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    # Create data directory in the project root
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    print(f"Downloading data for {len(symbols)} stocks from {start_date} to {end_date}...")
    
    download_stock_data(symbols, start_date, end_date, output_dir)
    
    print("\nDownload completed successfully!")

if __name__ == "__main__":
    main()
