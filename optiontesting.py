import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

_norm_cdf = stats.norm(0, 1).cdf
_norm_pdf = stats.norm(0, 1).pdf

def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_value(S, K, T, r, sigma):
    '''
    The fair value of a call option paying max(S-K, 0) at expiry, under the Black-scholes model,
    for an option with strike <K>, expiring in <T> years, under a fixed interest rate <r>,
    a stock volatility <sigma>, and when the current price of the underlying stock is <S>.
        
    Parameters
    ----------
    S : float
        The current value of the underlying stock.
    
    K : float
        The strike price of the option.
        
    T : float
        Time to expiry in years.
    
    r : float
        The fixed interest rate valid between now and expiry.
    
    sigma : float
        The volatility of the underlying stock process.
    
    Returns
    -------
    call_value : float
        The fair present value of the option.
        
    '''
    
    return S * _norm_cdf(_d1(S, K, T, r, sigma)) - K * np.exp(-r * T) * _norm_cdf(_d2(S, K, T, r, sigma))


def put_value(S, K, T, r, sigma):
    '''
    The fair value of a put option paying max(K-S, 0) at expiry, under the Black-scholes model,
    for an option with strike <K>, expiring in <T> years, under a fixed interest rate <r>,
    a stock volatility <sigma>, and when the current price of the underlying stock is <S>.
        
    Parameters
    ----------
    S : float
        The value of the underlying stock.
    
    K : float
        The strike price of the option.
        
    T : float
        Time to expiry in years.
    
    r : float
        The fixed interest rate valid between now and expiry.
    
    sigma : float
        The volatility of the underlying stock process.
    
    Returns
    -------
    put_value : float
        The fair present value of the option.
    '''
    
    return np.exp(-r * T) * K * _norm_cdf(-_d2(S, K, T, r, sigma)) - S * _norm_cdf(-_d1(S, K, T, r, sigma))


def call_delta(S, K, T, r, sigma):
    '''
    The delta, i.e. the first derivative of the option value with respect to the underlying, 
    of a call option paying max(S-K, 0) at expiry, under the Black-scholes model, for an option 
    with strike <K>, expiring in <T> years, under a fixed interest rate <r>, a stock 
    volatility <sigma>, and when the current price of the underlying stock is <S>.
        
    Parameters
    ----------
    S : float
        The value of the underlying stock.
    
    K : float
        The strike price of the option.
        
    T : float
        Time to expiry in years.
    
    r : float
        The fixed interest rate valid between now and expiry.
    
    sigma : float
        The volatility of the underlying stock process.
    
    Returns
    -------
    call_delta : float
        The fair present value of the option.   
    '''
    
    return _norm_cdf(_d1(S, K, T, r, sigma))


def put_delta(S, K, T, r, sigma):
    '''
    The delta, i.e. the first derivative of the option value with respect to the underlying, 
    of a put option paying max(K-S, 0) at expiry, under the Black-scholes model, for an option 
    with strike <K>, expiring in <T> years, under a fixed interest rate <r>, a stock 
    volatility <sigma>, and when the current price of the underlying stock is <S>.
        
    Parameters
    ----------
    S : float
        The value of the underlying stock.
    
    K : float
        The strike price of the option.
        
    T : float
        Time to expiry in years.
    
    r : float
        The fixed interest rate valid between now and expiry.
    
    sigma : float
        The volatility of the underlying stock process.
    
    Returns
    -------
    put_delta : float
        The fair present value of the option.   
    '''
    
    return call_delta(S, K, T, r, sigma) - 1


def call_vega(S, K, T, r, sigma):
    '''
    The vega, i.e. the derivative of the option value with respect to the volatility, 
    of a call option paying max(S-K, 0) at expiry, under the Black-scholes model, for an option 
    with strike <K>, expiring in <T> years, under a fixed interest rate <r>, a stock 
    volatility <sigma>, and when the current price of the underlying stock is <S>.
        
    Parameters
    ----------
    S : float
        The value of the underlying stock.
    
    K : float
        The strike price of the option.
        
    T : float
        Time to expiry in years.
    
    r : float
        The fixed interest rate valid between now and expiry.
    
    sigma : float
        The volatility of the underlying stock process.
    
    Returns
    -------
    call_delta : float
        The fair present value of the option.   
    '''
    
    return S * _norm_pdf(_d1(S, K, T, r, sigma)) * np.sqrt(T)


def put_vega(S, K, T, r, sigma):
    '''
    The vega, i.e. the derivative of the option value with respect to the volatility, 
    of a put option paying max(K-S, 0) at expiry, under the Black-scholes model, for an option 
    with strike <K>, expiring in <T> years, under a fixed interest rate <r>, a stock 
    volatility <sigma>, and when the current price of the underlying stock is <S>.
        
    Parameters
    ----------
    S : float
        The value of the underlying stock.
    
    K : float
        The strike price of the option.
        
    T : float
        Time to expiry in years.
    
    r : float
        The fixed interest rate valid between now and expiry.
    
    sigma : float
        The volatility of the underlying stock process.
    
    Returns
    -------
    call_delta : float
        The fair present value of the option.   
    '''
    
    return call_vega(S, K, T, r, sigma)


def read_data(filename):
    df = pd.read_csv(filename, index_col=0)

    time_to_expiry = df.filter(like='TimeToExpiry')

    stock = df.filter(like='Stock')
    stock.columns = [stock.columns.str[-5:], stock.columns.str[:-6]]

    options = pd.concat((df.filter(like='-P'), df.filter(like='-C')), axis=1)
    options.columns = [options.columns.str[-3:], options.columns.str[:-4]]

    market_data = pd.concat((stock, options), axis=1)

    return time_to_expiry, market_data

filename = 'Options Arbitrage.csv'
time_to_expiry, market_data = read_data(filename)

instrument_names = list(market_data.columns.get_level_values(0).unique())
print(instrument_names)

option_names = instrument_names[1:]
print(option_names)

market_data['TTE'] = time_to_expiry['TimeToExpiry']


timestamp = market_data.index


market_data = market_data.set_index('TTE')

short_call_values = {}
long_call_values = {}
long_put_values = {}
short_put_values = {}
short_call_deltas = {}
long_call_deltas = {}
long_put_deltas = {}
short_put_deltas = {}
option_values = {}
option_deltas = {}


r = 0
sigma = 0.20


for option in option_names:
    # Retrieve K from the Option
    K = int(option[-2:])

    if 'C' in option:
        short_call_values[option] = []
        long_call_values[option] = []
        short_call_deltas[option] = []
        long_call_deltas[option] = []

        # Forloop to calculate short/long call values and deltas
        for time, stock_value in market_data.iterrows():
            short_call_values[option].append(call_value(
                stock_value['Stock', 'AskPrice'], K, time, r, sigma))
            long_call_values[option].append(call_value(
                stock_value['Stock', 'BidPrice'], K, time, r, sigma))
            long_call_deltas[option].append(call_delta(
                stock_value['Stock', 'BidPrice'], K, time, r, sigma))
            short_call_deltas[option].append(-call_delta(
                stock_value['Stock', 'AskPrice'], K, time, r, sigma))

        option_values['Short Call', option] = short_call_values[option]
        option_values['Long Call', option] = long_call_values[option]
        option_deltas['Short Call', option] = short_call_deltas[option]
        option_deltas['Long Call', option] = long_call_deltas[option]

    if 'P' in option:
        long_put_values[option] = []
        short_put_values[option] = []
        long_put_deltas[option] = []
        short_put_deltas[option] = []

        # Forloop to calculate short/long put values and deltas
        for time, stock_value in market_data.iterrows():
            long_put_values[option].append(
                put_value(stock_value['Stock', 'AskPrice'], K, time, r, sigma))
            short_put_values[option].append(
                put_value(stock_value['Stock', 'BidPrice'], K, time, r, sigma))
            long_put_deltas[option].append(
                put_delta(stock_value['Stock', 'AskPrice'], K, time, r, sigma))
            short_put_deltas[option].append(-put_delta(
                stock_value['Stock', 'BidPrice'], K, time, r, sigma))

        option_values['Long Put', option] = long_put_values[option]
        option_values['Short Put', option] = short_put_values[option]
        option_deltas['Long Put', option] = long_put_deltas[option]
        option_deltas['Short Put', option] = short_put_deltas[option]


option_values = pd.DataFrame(option_values, index=market_data.index)
option_deltas = pd.DataFrame(option_deltas, index=market_data.index)

# Sort the DataFrames
option_values = option_values.reindex(sorted(option_values.columns), axis=1)
option_deltas = option_deltas.reindex(sorted(option_deltas.columns), axis=1)

# Rounding
option_values = round(option_values, 2)




for option in option_names:
    if "C" in option:
        market_data[option,
                    'Expected AskPrice'] = option_values['Short Call', option]
        market_data[option,
                    'Expected BidPrice'] = option_values['Long Call', option]
        market_data[option,
                    'Delta Short'] = option_deltas['Short Call', option].values
        market_data[option,
                    'Delta Long'] = option_deltas['Long Call', option].values

    elif "P" in option:
        market_data[option,
                    'Expected AskPrice'] = option_values['Short Put', option]
        market_data[option,
                    'Expected BidPrice'] = option_values['Long Put', option]
        market_data[option,
                    'Delta Short'] = option_deltas['Short Put', option].values
        market_data[option,
                    'Delta Long'] = option_deltas['Long Put', option].values

# Sort Columns
market_data = market_data.reindex(sorted(market_data.columns), axis=1)

def option_opportunities(option):
    '''
    This function gives arbitrage opportunities based on whether the price
    of the option is too high or too low. The results are used to 'eyeball' 
    if our final results match what this function displays. This works for 
    all Calls and Puts.
    '''
    if "C" in option or "P" in option:
        expected1 = market_data[option][(market_data[option, 'BidPrice'] - market_data[option,
                                                                                       'Expected AskPrice']) >= 0.10].drop('Expected BidPrice', axis=1)
        expected2 = market_data[option][(market_data[option, 'Expected BidPrice'] -
                                         market_data[option, 'AskPrice']) >= 0.10].drop('Expected AskPrice', axis=1)

    print('BidPrice is at least 0.10 higher than Expected AskPrice for Option ' + option)

    print('AskPrice is at least 0.10 lower than Expected BidPrice for Option ' + option)

    print('The amount of trades are', len(expected1) + len(expected2))

option_opportunities('C80')

trades = {('Timestamp', ''): timestamp,
          ('Time to Expiry', ''): market_data.index}

# Forloop that adds columns for the Call/Put Positions and Deltas
# Global function is a changing variable name based on the option
# For option C60 it will create a variable named positions_call_C60
for option in option_names:

    if 'C' in option:
        trades['Call Position', option] = []
        trades['Call Delta', option] = []
        globals()['positions_call_' + option] = 0

    if 'P' in option:
        trades['Put Position', option] = []
        trades['Put Delta', option] = []
        globals()['positions_put_' + option] = 0

for time, data in market_data.iterrows():

    max_delta = min(data['Stock', 'AskVolume'], data['Stock', 'BidVolume'])

    # Forloop over the option_names with conditions
    # if-statements if Call or Put + if Short/Long in Call or Put
    for option in option_names:

        if 'C' in option:

            # Short Call
            if (data[option, 'BidPrice'] - data[option, 'Expected AskPrice']) >= 0.10:
                short_call_volume = data[option, 'BidVolume']
                long_call_volume = 0

            # Long Call
            elif (data[option, 'Expected BidPrice'] - data[option, 'AskPrice']) >= 0.10:
                long_call_volume = data[option, 'AskVolume']
                short_call_volume = 0

            else:
                long_call_volume = short_call_volume = 0

            call_trade = long_call_volume - short_call_volume

            # Define variable, as set earlier. Note the first position is set to zero otherwise
            # One would get an error here since the variable is then not yet defined.
            globals()['positions_call_' + option] = call_trade + \
                globals()['positions_call_' + option]

            # Add Positions (cumulative)
            trades['Call Position', option].append(
                globals()['positions_call_' + option])

            if globals()['positions_call_' + option] >= 0:
                long_call_delta = data[option, 'Delta Long']
                short_call_delta = 0

            elif globals()['positions_call_' + option] < 0:
                short_call_delta = data[option, 'Delta Short']
                long_call_delta = 0

            # Add Deltas (cumulative)
            trades['Call Delta', option].append(
                abs(globals()['positions_call_' + option]) * (long_call_delta + short_call_delta))

        if 'P' in option:

            # Short Put
            if (data[option, 'BidPrice'] - data[option, 'Expected AskPrice']) >= 0.10:
                short_put_volume = data[option, 'BidVolume']
                long_put_volume = 0

            # Long Put
            elif (data[option, 'Expected BidPrice'] - data[option, 'AskPrice']) >= 0.10:
                long_put_volume = data[option, 'AskVolume']
                short_put_volume = 0

            else:
                long_put_volume = short_put_volume = 0

            put_trade = long_put_volume - short_put_volume

            globals()['positions_put_' + option] = put_trade + \
                globals()['positions_put_' + option]

            trades['Put Position', option].append(
                globals()['positions_put_' + option])

            if globals()['positions_put_' + option] >= 0:
                long_put_delta = data[option, 'Delta Long']
                short_put_delta = 0

            elif globals()['positions_put_' + option] < 0:
                short_put_delta = data[option, 'Delta Short']
                long_put_delta = 0

            trades['Put Delta', option].append(
                abs(globals()['positions_put_' + option]) * (long_put_delta + short_put_delta))

trades = pd.DataFrame(trades).set_index('Timestamp')

# Sort Columns
trades = trades.reindex(sorted(trades.columns), axis=1)

# Calculate Total Option Delta (based on sorted columns)
trades['Total Option Delta', ''] = np.sum(
    trades['Call Delta'], axis=1) + np.sum(trades['Put Delta'], axis=1)

# Calculate Cumulative Stock Position (floored if positive, ceiled if negative)
trades['Stock Position', 'Stock'] = -np.where(trades['Total Option Delta', ''] >= 0, np.floor(
    trades['Total Option Delta', '']), np.ceil(trades['Total Option Delta', '']))

# Calculate remaining option delta (that remains unhedged)
# This delta is included in the Total Option Delta again which ensures
# It always remains below zero
trades['Remaining Option Delta', ''] = trades['Total Option Delta',
                                              ''] + trades['Stock Position', 'Stock']

# Show DataFrame
trades.tail()


trades_diff = trades.diff()[1:].drop(
    ['Call Delta', 'Put Delta', 'Time to Expiry', 'Total Option Delta', 'Remaining Option Delta'], axis=1)

# Drop the 'Call Position','Put Position' and 'Stock Position' top level
# Makes forlooping easier
trades_diff.columns = trades_diff.columns.droplevel(level=0)

# Since positions are not neccesarily zero at the last timestamp, final positions are calculated to be able to valuate these
final_positions = trades[-1:].drop(['Call Delta', 'Put Delta', 'Time to Expiry',
                                    'Total Option Delta', 'Remaining Option Delta'], axis=1)

final_positions.columns = final_positions.columns.droplevel(level=0)

# Show DataFrames
print('Actual Trades/Volumes')

print("Final Positions that we currently 'own'")


market_data['Timestamp'] = timestamp
market_data = market_data.set_index('Timestamp')


cashflow_dataframe = pd.DataFrame(index=market_data.index[1:])

# Forloop on all instruments (including stock) to calculate PnL
for instrument in instrument_names:

    Instrument_AskPrice = market_data[instrument, 'AskPrice'][1:]
    Instrument_BidPrice = market_data[instrument, 'BidPrice'][1:]

    cashflow_dataframe[instrument] = np.where(trades_diff[instrument] >= 0,
                                              trades_diff[instrument] * -
                                              Instrument_AskPrice,
                                              trades_diff[instrument] * -Instrument_BidPrice)


total_cashflow = cashflow_dataframe.sum().sum()

print('The total Cashflow is: €', round(total_cashflow, 2))


cashflow_cumulative = {column: cashflow_dataframe[column].cumsum() for column in cashflow_dataframe.columns}


cashflow_cumulative = pd.DataFrame(cashflow_cumulative)

# Show Cumulative Cashflow


# Checking for Match
print('This number should match the above number: €',
      round(cashflow_cumulative[-1:].sum().sum(), 2))


trades_minimal = trades.drop(['Call Delta', 'Put Delta', 'Time to Expiry', 'Total Option Delta',
                              'Remaining Option Delta'], axis=1)

trades_minimal.columns = trades_minimal.columns.droplevel(level=0)

# Create Dataframe with market_data as index
valuation_dataframe = pd.DataFrame(index=market_data.index)

# Forloop to calculate valuations on every timestamp
for instrument in instrument_names:

    if 'C' in instrument:

        Instrument_AskPrice = market_data[instrument, 'AskPrice']
        Instrument_BidPrice = market_data[instrument, 'BidPrice']

        valuation_dataframe[instrument] = np.where(trades_minimal[instrument] > 0,
                                                   trades_minimal[instrument] *
                                                   Instrument_BidPrice,
                                                   trades_minimal[instrument] * Instrument_AskPrice)

    if 'P' in instrument:

        Instrument_AskPrice = market_data[instrument, 'AskPrice']
        Instrument_BidPrice = market_data[instrument, 'BidPrice']

        valuation_dataframe[instrument] = np.where(trades_minimal[instrument] > 0,
                                                   trades_minimal[instrument] *
                                                   Instrument_BidPrice,
                                                   trades_minimal[instrument] * Instrument_AskPrice)

    if 'S' in instrument:

        Instrument_AskPrice = market_data[instrument, 'AskPrice']
        Instrument_BidPrice = market_data[instrument, 'BidPrice']

        valuation_dataframe[instrument] = np.where(trades_minimal[instrument] > 0,
                                                   trades_minimal[instrument] *
                                                   Instrument_BidPrice,
                                                   trades_minimal[instrument] * Instrument_AskPrice)

total_valuation = valuation_dataframe[-1:].sum().sum()

print("Total valuation of our Position is currently: €", round(total_valuation, 2))


blackscholes_dataframe = {}

# Create Columns based on Option Names
for option in option_names:

    if 'C' in option:
        blackscholes_dataframe[option] = []

    if 'P' in option:
        blackscholes_dataframe[option] = []

# Forloop that calculates the margins and thus profits
for time, data in market_data.iterrows():

    for option in option_names:

        if "C" in option or "P" in option:
            margin1 = data[option, 'BidPrice'] - data[option, 'Expected AskPrice']
            margin2 = data[option, 'Expected BidPrice'] - data[option, 'AskPrice']

            if margin1 > 0.10:
                blackscholes_dataframe[option].append(margin1)

            elif margin2 > 0.10:
                blackscholes_dataframe[option].append(margin2)

            else:
                blackscholes_dataframe[option].append(0)

# Create DataFrame with index of market_data
blackscholes_dataframe = pd.DataFrame(blackscholes_dataframe, index=market_data.index)

# Calculate Black_Scholes Profit
total_blackscholes = (abs(trades_diff).drop('Stock', axis=1) * blackscholes_dataframe).sum().sum()
print('The total profit from Black Scholes is: €',round(total_blackscholes,2))

blackscholes_dataframe = pd.DataFrame(
    abs(trades_diff).drop('Stock', axis=1) * blackscholes_dataframe)

# Cumulative Blackscholes
blackscholes_cumulative = {column: blackscholes_dataframe[column].cumsum() for column in blackscholes_dataframe.columns}


blackscholes_cumulative = pd.DataFrame(blackscholes_cumulative)

# Show Dataframe
blackscholes_cumulative.tail()

print('The total profit generated from the Option Arbitrage strategy is: €',
      round(total_cashflow + total_valuation + total_blackscholes, 2))


