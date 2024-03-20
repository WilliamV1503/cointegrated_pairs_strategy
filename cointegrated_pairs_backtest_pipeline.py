import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
import pandas as pd
import yfinance as yf
import itertools
import statsmodels.api as sm
import random
import os 
import warnings


def calculate_returns(prices):

    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    return returns


def correlation_of_returns(asset1_prices, asset2_prices, threshold):

    asset1_returns = calculate_returns(asset1_prices)
    asset2_returns = calculate_returns(asset2_prices)
    
    correlation = np.corrcoef(asset1_returns, asset2_returns)[0, 1]

    return abs(correlation) > threshold


def calculate_rolling_beta(stock1_prices, stock2_prices, lookback_period):

    stock1_returns = stock1_prices.pct_change()
    stock2_returns = stock2_prices.pct_change()
    rolling_betas = []

    for i in range(lookback_period, len(stock1_prices)):

        cov = stock1_returns[i - lookback_period:i].cov(stock2_returns[i - lookback_period:i])
        var = stock2_returns[i - lookback_period:i].var()
        rolling_beta = cov / var
        rolling_betas.append(rolling_beta)
    
    rolling_beta_series = pd.Series(rolling_betas, index=stock1_prices.index[lookback_period:])

    return rolling_beta_series


def cointegration_test(y, x):
    
    #check for unit root in each series
    y_adf_pvalue = adfuller(y)[1]
    x_adf_pvalue = adfuller(x)[1]

    if y_adf_pvalue <= 0.05 and x_adf_pvalue <= 0.05:
        return False, -1
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    
    #run the augmented dickey fuller test on the residuals
    result = adfuller(residuals)
    p_value = result[1]

    if p_value < 0.05:
        return True, p_value
    else:
        return False, -1

#calculate spread between cointegrated stocks
def cointegration_spread(p1_series, p2_series, lookback_period):

    beta = calculate_rolling_beta(p1_series,p2_series,lookback_period)
    spread = p1_series.iloc[lookback_period:] - beta * p2_series.iloc[lookback_period:]

    return spread


#standardization calculated on rolling basis to avoid lookahead bias
def standardized_cointegration_spread(spread_series,lookback_period):
    
    mean = spread_series.rolling(lookback_period).mean()
    std = spread_series.rolling(lookback_period).std()
    standardized_spread = (spread_series - mean)/std
    return standardized_spread


def signal_labelling(spread_series_df,z_enter,z_exit,z_stoploss,days_limit):
    #spread_series_df -> standardized spread time series df
    #z_enter -> abs(z-score) at which to enter trade
    #z_exit -> abs(z-score) at which to exit trade
    #z_stoploss -> abs(z-score) at which to exit trade if it heads opposite of z_exit
    #days_limit -> max number of elapsed days to remain in any trade    

    #add column for action, filled with 0's
    spread_series_df.loc[:, 'action'] = pd.Series([0] * len(spread_series_df), index=spread_series_df.index)
    spread_series_df.loc[:, 'elapsed_trade_days'] = pd.Series([0] * len(spread_series_df), index=spread_series_df.index)
    
    currently_in_position = False
    cooldown = False
    days_counter = 0
    prev_key = 1

    #direction drxn variable is negative if z_enter < z_exit (ie dispersion strategy), thereby flipping inequality conditions
    drxn = 1
    if z_enter < z_exit:
        drxn *= -1
    
    for idx,row in spread_series_df.iterrows():
        if (days_counter > 21):
            print("here")
        if (drxn) * abs(row["standardized_spread"]) >= (drxn) * z_enter:

            #if at or beyond z_enter threshold, either in position or not
            if currently_in_position:
                spread_series_df.loc[idx, "action"] = prev_key
                #if stop_loss hit, exit current position
                if (drxn) * abs(row["standardized_spread"]) >= (drxn) * z_stoploss:
                    currently_in_position = False
                    days_counter+=1
                    spread_series_df.loc[idx, "elapsed_trade_days"] = days_counter
                    days_counter = 0
                else:
                    days_counter += 1
                    spread_series_df.loc[idx, "elapsed_trade_days"] = days_counter
                    #if day_limit hit, exit current position
                    if days_counter >= days_limit:
                        currently_in_position = False
                        cooldown = True
                        spread_series_df.loc[idx, "elapsed_trade_days"] = days_counter
                        days_counter = 0

            else:
                #if first action in period, create new unique identifier for that trade period 
                if cooldown==False:
                    prev_key = random.uniform(1,100)
                    if row["standardized_spread"]<0:
                        prev_key = -1 * prev_key
                    spread_series_df.loc[idx, "action"] = prev_key
                    currently_in_position = True
                            
        else:

            #if not within z_enter threshold, either in position or not
            if currently_in_position:
                #if z_exit hit or day_limit hit, exit current position
                if (drxn) * abs(row["standardized_spread"]) <= (drxn) * z_exit or days_counter >= days_limit:
                    currently_in_position = False
                    spread_series_df.loc[idx, "elapsed_trade_days"] = days_counter
                    days_counter = 0
                else:
                    #set action equal to the current trade period's unique identifier
                    spread_series_df.loc[idx, "action"] = prev_key
                    days_counter += 1
                    spread_series_df.loc[idx, "elapsed_trade_days"] = days_counter

            else:
                #if trade was exited when day_limit was hit beyond z_enter threshold, cooldown is reset once z_enter is crossed again
                if cooldown == True:
                    cooldown = False

    return spread_series_df


def daily_returns(signal_labelled_df):

    #calcluate cumulative returns per day for each asset corresponding to initial bid/ask price and current mid price
    
    if signal_labelled_df['action'].iloc[0] > 0:
        signal_labelled_df['return_asset_1'] = (signal_labelled_df['p1_mid'] - signal_labelled_df['p1_ask'].iloc[0])/signal_labelled_df['p1_ask'].iloc[0]
        signal_labelled_df['return_asset_2'] = (signal_labelled_df['p2_bid'].iloc[0] - signal_labelled_df['p2_mid'])/signal_labelled_df['p2_bid'].iloc[0]

    elif signal_labelled_df['action'].iloc[0] < 0:
        signal_labelled_df['return_asset_1'] = (signal_labelled_df['p1_ask'].iloc[0] - signal_labelled_df['p1_mid'])/signal_labelled_df['p1_ask'].iloc[0]
        signal_labelled_df['return_asset_2'] = (signal_labelled_df['p2_mid'] - signal_labelled_df['p2_ask'].iloc[0])/signal_labelled_df['p2_ask'].iloc[0]

    #adjust return of asset_2 which was bought in proportion to asset_1 by factor of beta (on date bought)
    correlation_beta = signal_labelled_df['correlation_beta'].iloc[0]
    signal_labelled_df['adjusted_asset_2'] = signal_labelled_df['return_asset_2'] * correlation_beta
    signal_labelled_df['total_return'] = signal_labelled_df['adjusted_asset_2'] + signal_labelled_df['return_asset_1']

    first_date = signal_labelled_df["date"].min()
    last_date = signal_labelled_df["date"].max()
    return_ = signal_labelled_df['total_return'].iloc[-1]
    pair = signal_labelled_df['pair'].iloc[0]
    sector = signal_labelled_df['sector'].iloc[0]
    elapsed_trade_days = signal_labelled_df["elapsed_trade_days"].iloc[-1]
    
    new_df = pd.DataFrame(data={'pair':[0],'f_date':[0],'l_date':[0],'return':[0]})
    new_df['return'] = return_
    new_df['f_date'] = first_date
    new_df['l_date'] = last_date
    new_df['pair'] = pair
    new_df["sector"] = sector
    new_df["correlation_beta"] = correlation_beta
    new_df["elapsed_trade_days"] = elapsed_trade_days
    new_df["elapsed_calendar_days"] = (new_df["l_date"].apply(lambda x: pd.to_datetime(x)) - new_df["f_date"].apply(lambda x: pd.to_datetime(x))).apply(lambda x: x.days)

    return new_df

def ticker_paths_by_year(year_list):
    
    dict_of_paths = {}
    for year in year_list:
        dict_of_paths[year]=[]
    lst_of_dir = os.listdir("C:\\Users\\wvill\\OneDrive\\FQE\\data2\\final_file_data\\")
    for file in lst_of_dir:
        if file.endswith('.csv'):
            sp = file.split('_')
            year = sp[1].split('.')[0]
            if year in year_list:
                dict_of_paths[year].append(file)

    return dict_of_paths

def ticker_sets_by_year(dict_of_paths):
    
    ticker_sets = {}
    for year, filenames in dict_of_paths.items():
        ticker_set = set()
        for filename in filenames:
            ticker = filename.split("_")[0]
            ticker_set.add(ticker)
        ticker_sets[year] = ticker_set

    return ticker_sets

#keep file paths of tickers in data set that were not delisted up to the reference year
def surviving_ticker_paths(dict_of_sets, dict_of_paths, reference_year):
    
    intersection_set = dict_of_sets[reference_year]
    
    for year, set in dict_of_sets.items():
        intersection_set = intersection_set.intersection(set)
    
    filtered_dict_of_paths = {}
    
    for year, filenames in dict_of_paths.items():
        filtered_filenames = [filename for filename in filenames if filename.split("_")[0] in intersection_set]
        filtered_dict_of_paths[year] = filtered_filenames

    return filtered_dict_of_paths
    

def clean_sptm_csv():
    
    #keeps columns containing constituents of december of each year
    sector_list = ["communicationservices","consumerdiscretionary","consumerstaples","energy",
         "financials","healthcare","industrials","informationtechnology","materials","realestate","utilities"]
    for sector in sector_list: 
        holdings_df = pd.read_csv(f"sptm_by_industry\\sptm_{sector}.csv")
        holdings_df = holdings_df[["Exchange:Ticker"]+[i for i in holdings_df.columns if "Dec" in i]]
        for i in holdings_df.columns[1:]:
            temp = i.split("-")[2]
            temp = temp.split()[0]
            holdings_df.rename({i:temp},axis=1,inplace=True)
        holdings_df=holdings_df[holdings_df["Exchange:Ticker"].notna()]
        holdings_df["Exchange:Ticker"]=holdings_df["Exchange:Ticker"].apply(lambda x: x if "Inactive" not in x else np.nan)
        holdings_df=holdings_df[holdings_df["Exchange:Ticker"].notna()]
        holdings_df["Ticker"]=holdings_df["Exchange:Ticker"].apply(lambda x: x.split(":")[-1])
        holdings_df = holdings_df.replace("  -  ",np.nan)
        holdings_df.to_csv(f"sptm_by_industry\\sptm_clean_{sector}.csv")
        del holdings_df
        
    
def sptm_constituents(dict_of_tickers):

    dict_of_constituents = {}
    for year in dict_of_tickers.keys():
        
        constituents_by_sector = {"communicationservices":[],"consumerdiscretionary":[],"consumerstaples":[],"energy":[],
             "financials":[],"healthcare":[],"industrials":[],"informationtechnology":[],"materials":[],"realestate":[],"utilities":[]}
        for sector in constituents_by_sector.keys():
    
            #sorts ticker paths into industry lists if they were sptm constituents
            holdings_df = pd.read_csv(f"sptm_by_industry\\sptm_clean_{sector}.csv")
            temp = holdings_df[str(int(year)-1)]
            temp_index = temp[temp.notna()].index
            holdings_df = holdings_df.loc[temp_index]["Ticker"].unique()
            for tick in holdings_df:
                temp_filename = f"{tick}_{year}.csv"
                if temp_filename in filtered_dict_of_paths[year]:
                    constituents_by_sector[sector].append(temp_filename)
                    
        dict_of_constituents[year]=constituents_by_sector
        
    return dict_of_constituents

    
def sptm_constituents(dict_of_tickers):

    dict_of_constituents = {}
    for year in dict_of_tickers.keys():
        
        constituents_by_sector = {"communicationservices":[],"consumerdiscretionary":[],"consumerstaples":[],"energy":[],
             "financials":[],"healthcare":[],"industrials":[],"informationtechnology":[],"materials":[],"realestate":[],"utilities":[]}
        for sector in constituents_by_sector.keys():
    
            #sorts ticker paths into industry lists if they were sptm constituents
            holdings_df = pd.read_csv(f"sptm_by_industry\\sptm_clean_{sector}.csv")
            temp = holdings_df[str(int(year)-1)]
            temp_index = temp[temp.notna()].index
            holdings_df = holdings_df.loc[temp_index]["Ticker"].unique()
            for tick in holdings_df:
                temp_filename = f"{tick}_{year}.csv"
                if temp_filename in filtered_dict_of_paths[year]:
                    constituents_by_sector[sector].append(temp_filename)
                    
        dict_of_constituents[year]=constituents_by_sector
        
    return dict_of_constituents


def find_pairs_by_industry_year(dict_of_constituents):

    #old column names from Millisecond Data
    old_cols = ['Date', 'ticker', 'ClosePrice', 'OpenPrice', 'nbb_4pm', 'nbo_4pm', 'total_vol']
    main_cols = ['date', 'ticker', 'Close', 'Open', 'bid', 'ask', 'Volume']

    dict_of_pairs = {}
    #pairs found in year t will be traded in year t+1
    for year in list(dict_of_constituents.keys())[:-1]:

        pairs_by_sector = {"communicationservices":[],"consumerdiscretionary":[],"consumerstaples":[],"energy":[],
             "financials":[],"healthcare":[],"industrials":[],"informationtechnology":[],"materials":[],"realestate":[],"utilities":[]}
        for sector in pairs_by_sector.keys():
            
            print(f"looking for pairs in year {year} in sector {sector}")
            year_files = sorted(dict_of_constituents[year][sector])
            #checks all possible combinations of pairs
            for i in range(len(year_files)-1):
                    
                ticker_1 = year_files[i]
                print(f"looking for pairs for {ticker_1}")
                ticker_1_df = pd.read_csv(f'C:\\Users\\wvill\\OneDrive\\FQE\\data2\\final_file_data\\{ticker_1}')
                ticker_1_df = ticker_1_df.rename(columns={old_col: new_col for old_col, new_col in zip(old_cols, main_cols)})
                ticker_1_df = ticker_1_df[ticker_1_df['Close'].notna()].reset_index(drop=True)
                
                for j in range(i+1,len(year_files)):
                    
                    ticker_2 = year_files[j]
                    ticker_2_df = pd.read_csv(f'C:\\Users\\wvill\\OneDrive\\FQE\\data2\\final_file_data\\{ticker_2}')
                    ticker_2_df = ticker_2_df.rename(columns={old_col: new_col for old_col, new_col in zip(old_cols, main_cols)})
                    ticker_2_df = ticker_2_df[ticker_2_df['Close'].notna()].reset_index(drop=True)
                    ticker_2_df["date"] = pd.to_datetime(ticker_2_df["date"])
                    
                    ticker_1_df_temp = ticker_1_df.copy()
                    ticker_1_df_temp["date"] = pd.to_datetime(ticker_1_df_temp["date"])
    
                    #intersection of dates in ticker_1 and ticker_2 dfs
                    intersect_dates = set(ticker_1_df_temp['date']).intersection(set(ticker_2_df['date']))
    
                    ticker_1_df_temp = ticker_1_df_temp[ticker_1_df_temp['date'].isin(intersect_dates)]
                    ticker_2_df = ticker_2_df[ticker_2_df['date'].isin(intersect_dates)]
                    
                    ticker_1_df_temp = ticker_1_df_temp.reset_index(drop=True)
                    ticker_2_df = ticker_2_df.reset_index(drop=True)
                    
                    try:
                        #correlation test to further filter out pairs (refer to paper)
                        are_correlated = correlation_of_returns(ticker_1_df_temp["Close"],ticker_2_df["Close"],threshold=0.7)
                        if are_correlated:
                            print(f"{ticker_1},{ticker_2} passed correlation test")
                            are_cointegrated, p = cointegration_test(ticker_1_df_temp['Close'], ticker_2_df['Close'])
    
                            if are_cointegrated:
                                print(f"{(ticker_1,ticker_2)} passed cointegration test")
                                pairs_by_sector[sector].append([ticker_1, ticker_2])
                                print(f"pair added {(ticker_1,ticker_2)}")
                            else:
                                print(f"{(ticker_1,ticker_2)} failed cointegration test")

                    except Exception as e:
                        print(f"failed to perform test on {(ticker_1,ticker_2)}")
                        print(e)
                        continue
                        
        print(f"finished looking for pairs in year: {year}, sector: {sector}")
    
        temp_pairs_df = pd.DataFrame({sector: pairs_by_sector[sector]})
        temp_pairs_df.to_csv(f"pairs_found\\{year}_{sector}_pairs.csv", index=False)
    
        dict_of_pairs[year]=pairs_by_sector

    return dict_of_pairs

#create historical trade table use signal labelling function
def create_trade_table(dict_of_pairs,signal_label_args):

    z_exit = signal_label_args["z_exit"]
    z_enter = signal_label_args["z_enter"]
    z_stoploss = signal_label_args["z_stoploss"]

    condition_1 = (0 < z_exit) and (z_exit < z_enter) and (z_enter < z_stoploss)
    condition_2 = (0 < z_stoploss) and (z_stoploss < z_enter) and (z_enter < z_exit)

    if not (condition_1 or condition_2):
        print("signal generating criteria invalid: let 0 < z_exit < z_enter < z_stoploss or 0 < z_stoploss < z_enter < z_exit")

    else:
            
    
        #old column names from Millisecond Data
        old_cols = ['Date', 'ticker', 'ClosePrice', 'OpenPrice', 'nbb_4pm', 'nbo_4pm', 'total_vol']
        main_cols = ['date', 'ticker', 'Close', 'Open', 'bid', 'ask', 'Volume']
    
        trade_table_rows = []
        last_year = False
        #goes up to year i-2 in active_years; trades pairs found in year i-2 in years i-1 and i
        for year in list(dict_of_pairs.keys()):
     
            year_1_fwd = int(year)+1
            if year == list(dict_of_pairs.keys())[-1]: 
                last_year = True
            else:
                year_2_fwd = int(year)+2
            print("jere")
            for sector in list(dict_of_pairs[year].keys()):
    
                if last_year:
                    print(f"labelling trades in year: {year_1_fwd}, sector: {sector}")
                else:
                    print(f"labelling trades in years: {year_1_fwd} and {year_2_fwd}, sector: {sector}")
                    
                for files in dict_of_pairs[year][sector]:
                    
                    asset_1_file = files[0]
                    asset_2_file = files[1]
                    asset_1_ticker = asset_1_file.split('_')[0]
                    asset_2_ticker = asset_2_file.split('_')[0]
    
                    asset_1_file_1yfwd = f"{asset_1_ticker}_{year_1_fwd}.csv"
                    asset_2_file_1yfwd = f"{asset_2_ticker}_{year_1_fwd}.csv"
    
                    #df_i_j where i is asset number and j is forward year number
                    df_1_1 = pd.read_csv(f'C:\\Users\\wvill\\OneDrive\\FQE\\data2\\final_file_data\\{asset_1_file_1yfwd}')
                    df_2_1 = pd.read_csv(f'C:\\Users\\wvill\\OneDrive\\FQE\\data2\\final_file_data\\{asset_2_file_1yfwd}')
        
                    df_1_1 = df_1_1.rename(columns={old_col: new_col for old_col, new_col in zip(old_cols, main_cols)})
                    df_2_1 = df_2_1.rename(columns={old_col: new_col for old_col, new_col in zip(old_cols, main_cols)})
    
                    if not last_year:
                        
                        asset_1_file_2yfwd = f"{asset_1_ticker}_{year_2_fwd}.csv"
                        asset_2_file_2yfwd = f"{asset_2_ticker}_{year_2_fwd}.csv"
                        
                        df_1_2 = pd.read_csv(f'C:\\Users\\wvill\\OneDrive\\FQE\\data2\\final_file_data\\{asset_1_file_2yfwd}')
                        df_2_2 = pd.read_csv(f'C:\\Users\\wvill\\OneDrive\\FQE\\data2\\final_file_data\\{asset_2_file_2yfwd}')
    
                        df_1_2 = df_1_2.rename(columns={old_col: new_col for old_col, new_col in zip(old_cols, main_cols)})
                        df_2_2 = df_2_2.rename(columns={old_col: new_col for old_col, new_col in zip(old_cols, main_cols)})
                    
    
                    if last_year:
                        df_1 = df_1_1
                        df_2 = df_2_1
                    else:
                        df_1 = pd.concat([df_1_1,df_1_2])
                        df_2 = pd.concat([df_2_1,df_2_2])
    
                    
                    intersect_dates = set(df_1['date']).intersection(set(df_2['date']))
                    df_1 = df_1[df_1['date'].isin(intersect_dates)]
                    df_2 = df_2[df_2['date'].isin(intersect_dates)]
                    df_1 = df_1.reset_index(drop = True)
                    df_2 = df_2.reset_index(drop = True)
                    
                    df_1 = df_1[main_cols]
                    df_2 = df_2[main_cols]
                    
                    spread = cointegration_spread(df_1['Close'],df_2['Close'],60)
                    spread_standardized = standardized_cointegration_spread(spread,60)
                    pair = df_1['ticker'] + '/' + df_2['ticker']
                    
                    df_both = pd.DataFrame(data = {'date': df_1['date'],
                                                   'pair': pair,
                                                   'spread': spread,
                                                   'standardized_spread': spread_standardized,
                                                   'p1_mid': df_1['Close'],
                                                   'p1_ask': df_1['ask'],
                                                   'p1_bid': df_1['bid'],
                                                   'p2_mid': df_2['Close'],
                                                   'p2_ask': df_2['ask'],
                                                   'p2_bid': df_2['bid']})
            
                    df_both["correlation_beta"] = calculate_rolling_beta(df_both["p1_mid"],df_both["p2_mid"],60)
                    df_both["sector"] = sector
                    
                    signal_label_args["spread_series_df"] = df_both
                    signal_labelled_df = signal_labelling(**signal_label_args)
                    signal_labelled_df = signal_labelled_df[signal_labelled_df['action'] != 0].sort_values('date').reset_index(drop = True)
                    returns = signal_labelled_df.groupby('action').apply(lambda x: daily_returns(x)).reset_index(drop = True)
                
                    trade_table_rows.append(returns)
    
                if last_year:
                    print(f"finished labelling trades in year: {year_1_fwd}, sector: {sector}")
                else:
                    print(f"finished labelling trades in years: {year_1_fwd} and {year_2_fwd}, sector: {sector}")
                break
        #concatenate all rows of pairs and their corresponding information
        if len(trade_table_rows) != 0:
            df_all = pd.concat(trade_table_rows)
            if df_all.empty:
                print("no trades found")
                return None
            else:
                df_all = df_all.sort_values('f_date').reset_index(drop=  True)
                df_all.to_csv("trade_table_full.csv",index = False)
                return df_all
        else:
            print("no trades found")
            return None


def main():

    warnings.filterwarnings('ignore')
    active_years = [str(i) for i in range(2017,2024)]

    #clean_sptm_csv()

    dict_of_paths = ticker_paths_by_year(active_years)
    ticker_sets = ticker_sets_by_year(dict_of_paths)
    filtered_dict_of_paths = surviving_ticker_paths(ticker_sets,dict_of_paths,'2023')

    dict_of_constituents = sptm_constituents(filtered_dict_of_paths)

    #dict_of_pairs and trade_table take long time to create for large number of stocks
    dict_of_pairs = find_pairs_by_industry_year(dict_of_constituents)

    signal_labelling_args = {
        "spread_series_df": None,
        "z_enter": 2,
        "z_exit": 1,
        "z_stoploss": 1.2,
        "days_limit": 21
    }
    trade_table = create_trade_table(dict_of_pairs,signal_labelling_args)
    trade_table["return"].describe()
        

