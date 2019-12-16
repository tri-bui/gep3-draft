import numpy as np
import pandas as pd




def get_stats(df):
    
    '''
    Function:
        Build a table of summary statistics
        
    Input:
        df - Pandas dataframe
        
    Output:
        Pandas dataframe similar to the DataFrame.describe() method
    '''
    
    stats = pd.DataFrame(df.count(), columns=['count_'])
    stats['missing_'] = df.shape[0] - stats.count_
    stats['mean_'] = df.mean()
    stats['std_'] = df.std()
    stats['min_'] = df.min().map(lambda n: np.NaN if type(n) not in [int, float] else n)
    stats['max_'] = df.max().map(lambda n: np.NaN if type(n) not in [int, float] else n)
    stats['dtype_'] = df.dtypes
    return stats.T.fillna('-')




def reidx_site_time(df, site_col, time_col, tstart, tend):
    
    '''
    Function:
        Reindex dataframe to include a timestamp for every hour within the time interval
        
    Input:
        df - Pandas dataframe with a site column and time column
        site_col - name of column containing sites
        time_col - name of column containing timestamps
        tstart - first timestamp in the format '{month}/{day}/{year} hh:mm:ss'
        tend - last timestamp in the same format
        
    Output:
        Pandas dataframe with full timestamp
    '''
    
    sites = df[site_col].unique()
    frame = df.set_index([site_col, time_col])
    
    frame = frame.reindex(
        pd.MultiIndex.from_product([
            sites,
            pd.date_range(start=tstart, end=tend, freq='H')
        ])
    )
    
    frame.index.rename([site_col, time_col], inplace=True)
    return frame.reset_index()




def get_site(df, time_col, site_col, site_num):
    
    '''
    Function:
        Extract the data from 1 site as a time series
        
    Input:
        df - Pandas dataframe with a site column and time column
        time_col - name of column containing timestamps
        site_col - name of column containing sites
        site_num - site number to extract
        
    Output:
        Pandas dataframe with a datetime index
    '''
    
    return df[df[site_col] == site_num].drop(site_col, axis=1).set_index(time_col)




def locate_missing(df, site_col, time_col, pct=False):
    
    '''
    Function:
        Count missing values by site
    
    Input:
        df - Pandas dataframe with a site column and time column
        time_col - name of column containing timestamps
        site_col - name of column containing sites
        pct (optional) - boolean to specify whether or not to convert the output to percentages
    
    Output:
        Pandas dataframe displaying a matrix of missing values by site
    '''

    # # missing
    
    missing = df.groupby(site_col).count()
    
    for col in df.columns[2:]:
        missing[col] = missing.timestamp - missing[col]
    
    missing.columns = [col if col == time_col else f'missing_{col}' for col in missing.columns]
    
    # % missing
    
    pct_missing = missing.copy()
    
    for col in missing.columns[1:]:
        pct_missing[col] = round(missing[col] * 100 / missing.timestamp, 2)
        
    pct_missing.columns = [col if col == time_col else f'pct_{col}' for col in pct_missing.columns]
    
    return pct_missing if pct else missing
    
    
    

def fill_missing(df, site_col, cols_to_ffill, cols_to_interp_lin, cols_to_interp_cubic):
    
    '''
    Function:
        Fill missing values by site
        
    Input:
        df - Pandas dataframe with a site column
        site_col - name of column containing sites
        cols_to_ffill - list of columns to perform a simple forward fill and backward fill on
        cols_to_interp_lin - list of columns to perform linear interpolation on
        cols_to_interp_cubic - list of columns to perform cubic interpolation on
        
        Note: cols_to_interp_lin and cols_to_interp_cubic will also be forward-filled and backward-filled (after interpolation) to fill the beginning and end
        
    Output:
        Pandas dataframe with missing data filled
        
        Note: sites missing 100% of the values will not be filled
    '''
    
    for col in df.columns:
        
        if col in cols_to_ffill:
            df[col] = df.groupby(site_col)[col] \
                        .transform(lambda s: s.fillna(method='ffill') \
                                              .fillna(method='bfill'))
                    
        if col in cols_to_interp_lin:
            df[col] = df.groupby(site_col)[col] \
                        .transform(lambda s: s.interpolate('linear', limit_direction='both') \
                                              .fillna(method='ffill') \
                                              .fillna(method='bfill'))
            
        if col in cols_to_interp_cubic:
            df[col] = df.groupby(site_col)[col] \
                        .transform(lambda s: s.interpolate('cubic', limit_direction='both') \
                                              .fillna(method='ffill') \
                                              .fillna(method='bfill'))

    return df




def to_local_time(df, site_col, time_col, timezones):
    
    '''
    Function:
        Convert timestamps to local time
        
    Input:
        df - Pandas dataframe with a site column and time column
        site_col - name of column containing sites
        time_col - name of column containing timestamps
        timezones - list of timezone offsets
        
    Output:
        Pandas dataframe with local time
    '''
    
    offset = df[site_col].map(lambda s: np.timedelta64(timezones[s], 'h'))
    df[time_col] += offset
    return df




def convert_readings(df, site_col, meter_col, reading_col, site_num, meter_type, conversion):
    
    '''
    Function:
        Convert the meter reading units of a specified meter type in a specified site
        
    Input:
        df - Pandas dataframe with a site column, meter type column, and meter reading column
        site_col - name of column containing sites
        meter_col - name of column containing meter types
        reading_col - name of column containing meter readings
        site_num - site number to make conversions in
        meter_type - meter type to make conversions for
        conversion - string to specify unit conversions in the format '{unit1}_to_{unit2}'
        
    Output:
        Pandas dataframe with converted units for a given meter type in a given site
    '''
    
    # Conversion multipliers
    kbtu_to_kwh = 0.2931
    kwh_to_kbtu = 3.4118
    kbtu_to_ton = 0.0833
    ton_to_kbtu = 12
    
    mult = eval(conversion)
    df.loc[(df[site_col] == site_num) & (df[meter_col] == meter_type), reading_col] *= mult
    return df




