#!/usr/bin/env python
import os, sys
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
import fix_yahoo_finance as yf

yf.pdr_override()

sys.path.append('../python')
from finance_utils import interpolate_missing, winsorize, compute_signal, covariance_matrix

def portfolio101(tickers, start, end):

    #Download data
    print "Downloading financial data..."
    data = pdr.get_data_yahoo(tickers=tickers, start=start, end=end)
    data.columns = data.columns.swaplevel(0, 1)
    data.sortlevel(0, axis=1, inplace=True)

    #Fill missing data with average from previous 5 days, if needed
    print "Interpolating missing data"
    for t in tickers:
        data.update( pd.DataFrame( 
                { t: 
                  { 'Adj Close': interpolate_missing(data[t]['Adj Close'].values, mean_window=5) } 
                  } 
                ) )
        
    #Computing signal (return on average from previous 20 days)
    print "Computing signal" 
    for t in tickers:
        data.loc[:,(t,"Returns")] = compute_signal(data[t]['Adj Close'].values, mean_window=20)

    #Perform Winsorization to remove outliers and keep [5%-95%] of data
    print "Perform Winsorization"
    for t in tickers:
        data.update( pd.DataFrame(
                { t:
                      { 'Returns': winsorize(data[t]['Returns'].values, winsor=5) }
                  }
                ) )

    #Compute covariance matrix of daily returns
    print "Computing variance/covariance"
    idx = []
    for t in tickers:
        idx.append( (t,"Returns") )
    sgn = data[idx]
    cov = covariance_matrix( sgn )    

if __name__ == "__main__":
    tickers = ['AMZN','AAPL','GOOG','T','KO','MCD','MAR','NKE','MA','EBAY']
    start = '2007-01-01'
    end = '2017-01-01'
    portfolio101(tickers, start, end)
