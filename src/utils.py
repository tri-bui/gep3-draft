import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def reduce_mem(df):
    
    '''
    Function:
        Recast column data types to reduce memory usage
        
    Input:
        Pandas dataframe
        
    Output:
        Pandas dataframe with reduced memory usage
    '''
    
    uint8_lim = 2 ** 8
    uint16_lim = 2 ** 16
    uint32_lim = 2 ** 32

    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            if df[col].max() < uint8_lim and df[col].min() >= 0:
                df[col] = df[col].astype('uint8')
            elif df[col].max() < uint16_lim and df[col].min() >= 0:
                df[col] = df[col].astype('uint16')
            elif df[col].max() < uint32_lim and df[col].min() >= 0:
                df[col] = df[col].astype('uint32')
                
    return df




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




def reidx_site_time(df, tstart, tend, site_col='site_id', time_col='timestamp'):
    
    '''
    Function:
        Reindex dataframe to include a timestamp for every hour within the time interval
        
    Input:
        df - Pandas dataframe with a site column and time column
        tstart - first timestamp in the format '{month}/{day}/{year} hh:mm:ss'
        tend - last timestamp in the same format
        site_col (optional) - name of column containing sites
        time_col (optional) - name of column containing timestamps
        
        Note: pass in site_col and time_col if different from defaults
        
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




def get_site(df, site_num, time_idx=False, site_col='site_id', time_col='timestamp'):
    
    '''
    Function:
        Extract the data from 1 site
        
    Input:
        df - Pandas dataframe with a site column and time column
        site_num - site number to extract
        time_idx (optional) - boolean to indicate weather or not to set the time as the index
        site_col (optional) - name of column containing sites
        time_col (optional) - name of column containing timestamps
        
        Note: pass in site_col and time_col if different from defaults
        
    Output:
        Pandas dataframe with data from 1 site
    '''
    
    df = df[df[site_col] == site_num].drop(site_col, axis=1)
    if time_idx:
        df.set_index(time_col, inplace=True)
    return df




def locate_missing(df, pct=False, site_col='site_id', time_col='timestamp'):
    
    '''
    Function:
        Count missing values by site
    
    Input:
        df - Pandas dataframe with a site column and time column
        pct (optional) - boolean to indicate whether or not to convert the output to percentages
        site_col (optional) - name of column containing sites
        time_col (optional) - name of column containing timestamps
        
        Note: pass in site_col and time_col if different from defaults
    
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
    
    
    

def fill_missing(df, cols_to_ffill, cols_to_interp_lin, cols_to_interp_cubic, site_col='site_id'):
    
    '''
    Function:
        Fill missing values by site
        
    Input:
        df - Pandas dataframe with a site column
        site_col - name of column containing sites
        cols_to_ffill - list of columns to perform a simple forward fill and backward fill on
        cols_to_interp_lin - list of columns to perform linear interpolation on
        cols_to_interp_cubic - list of columns to perform cubic interpolation on
        site_col (optional) - name of column containing sites
        
        Note: cols_to_interp_lin and cols_to_interp_cubic will also be forward-filled and backward-filled (after interpolation) to fill the beginning and end
        Note: pass in site_col if different from default
        
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




def to_local_time(df, timezones, site_col='site_id', time_col='timestamp'):
    
    '''
    Function:
        Convert timestamps to local time
        
    Input:
        df - Pandas dataframe with a site column and time column
        timezones - list of timezone offsets
        site_col (optional) - name of column containing sites
        time_col (optional) - name of column containing timestamps
        
        Note: pass in site_col and time_col if different from defaults
        
    Output:
        Pandas dataframe with local time
    '''
    
    offset = df[site_col].map(lambda s: np.timedelta64(timezones[s], 'h'))
    df[time_col] += offset
    return df




def convert_readings(df, site_num, meter_type, conversion, site_col='site_id', meter_col='meter', reading_col='meter_reading'):
    
    '''
    Function:
        Convert the meter reading units of a specified meter type in a specified site
        
    Input:
        df - Pandas dataframe with a site column, meter type column, and meter reading column
        site_num - site number to make conversions in
        meter_type - meter type to make conversions for
        conversion - string to specify unit conversions in the format '{unit1}_to_{unit2}'
        site_col (optional) - name of column containing sites
        meter_col (optional) - name of column containing meter types
        reading_col (optional) - name of column containing meter readings
        
        Note: pass in site_col, meter_col, and reading_col if different from defaults
        
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




def print_missing_readings(df, building_col='building_id', meter_col='meter', time_col='timestamp'):
    
    '''
    Function:
        Print the details of missing meter readings
        
    Input:
        df - Pandas dataframe with a building column, meter type column, and time column
        building_col (optional) - name of column containing buildings
        meter_col (optional) - name of column containing meter types
        time_col (optional) - name of column containing timestamps
        
        Note: pass in building_col, meter_col, and time_col if different from defaults
        
    Output:
        None
    '''
    
    types = ['electricity', 'chilledwater', 'steam', 'hotwater']
    
    by_bm = df.groupby([building_col, meter_col]).count().reset_index()
    m_count = by_bm[meter_col].value_counts()
    
    metr_m = by_bm[by_bm[time_col] != 8784]
    bldg_m = metr_m[building_col].nunique()
    type_m = metr_m[meter_col].value_counts()
    
    print(f'{bldg_m} different buildings ({bldg_m * 100 // by_bm[building_col].nunique()}%) have meters that are missing readings')
    print(f'A total of {metr_m.shape[0]} meters ({metr_m.shape[0] * 100 // by_bm.shape[0]}%) are missing readings\n')

    for i in range(type_m.shape[0]):
        print(f'{type_m[i]} {types[i]} meters ({type_m[i] * 100 // m_count[i]}%) are missing readings')
        
        
        
        
def plot_readings(df, buildings, freq=None, group=None, start=None, end=None, ticks=None, reverse=False, time_col='timestamp', building_col='building_id', meter_col='meter', reading_col='meter_reading'):
    
    '''
    Function:
        Plot readings from 1 or more of each type of meter
        
    Input:
        df - Pandas dataframe with a building column, meter type column, and time column
        buildings - an iterable of buildings
        freq (optional) - resampling frequency
        group (optional) - column or list of columns to group by
        start (optional) - the start index to slice buildings on
        end (optional) - the end index to slice buildings on
        ticks (optional) - a range of xtick locations
        reverse (optional) - a boolean to indicate whether or not to iterate through buildings first then meters
        time_col (optional) - name of column containing timestamps
        building_col (optional) - name of column containing buildings
        meter_col (optional) - name of column containing meter types
        reading_col (optional) - name of column containing meter readings
        
        Note: pass in freq if resampling or group if aggregating
        Note: pass in time_col, building_col, meter_col, and reading_col if different from defaults
        
    Output:
        None
    '''
    
    df = df.set_index(time_col)
    types = ['electricity', 'chilledwater', 'steam', 'hotwater']
    
    if reverse:
        for b in buildings:
            for m in df[df[building_col] == b][meter_col].unique():
                fig = plt.figure(figsize=(16, 4))
                bm = df[(df[building_col] == b) & (df[meter_col] == m)]
                bm.resample('d').mean()[reading_col].plot()
                plt.title(f'Building {b} ({types[m]} meter)')
                plt.ylabel('meter_reading')
    else:
        for m in range(4):
            for b in buildings[m][start:end]:
                fig = plt.figure(figsize=(16, 4))
                bm = df[(df[building_col] == b) & (df[meter_col] == m)]
                if freq:
                    bm = bm.resample(freq).mean()
                if group:
                    bm = bm.groupby(group).mean()
                bm[reading_col].plot(xticks=ticks)
                plt.title(f'Building {b} ({types[m]} meter)')
                plt.ylabel('meter_reading')

                
                
                
def show_elec_readings(df, by, idx='timestamp', vals='meter_reading', freq=None, legend_pos=(1, 1), legend_col=1, cols_to_sep=[], meter_col='meter', type_col='type'):
    
    '''
    Function:
        Plot electric meter readings by a specified feature
        
    Input:
        df - Pandas dataframe with 2 meter type columns (1 is integer-encoded)
        by - name of column to pivot to columns
        idx (optional) - name of column to set as index in pivot table
        vals (optional) - name of column to aggregate from for pivot table
        freq (optional) - resampling frequency
        legend_pos (optional) - a tuple to indicate the legend's anchor position
        legend_col (optional) - number of columns in legend
        cols_to_sep (optional) - list of columns to plot separately
        meter_col (optional) - name of column containing meter type integers
        type_col (optional) - name of column containing meter type strings
        
        Note: plot columns with a different scale separately for a better view
        Note: pass in time_col, building_col, meter_col, and reading_col if different 
        
    Output:
        Pandas dataframe of a pivot table
    '''
    
    elec = df.pivot_table(index=idx, columns=by, values=vals, aggfunc='mean')
    if freq:
        elec = elec.resample(freq).mean()
    
    elec.drop(cols_to_sep, axis=1).plot(figsize=(16, 6))
    plt.title('Electric meter readings')
    plt.ylabel('meter_reading')
    plt.legend(bbox_to_anchor=legend_pos, ncol=legend_col, fancybox=True)
    
    if cols_to_sep:
        elec[cols_to_sep].plot(figsize=(16, 6))
        plt.title('Electric meter readings')
        plt.ylabel('meter_reading')
        plt.legend(bbox_to_anchor=(1, 1), fancybox=True)
        
    return elec