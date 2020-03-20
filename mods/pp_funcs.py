import config
import numpy as np
import pandas as pd


def convert_to_local_time(df,
                          offset_list=config.tz_offsets,
                          site_var='site_id',
                          time_var='timestamp'):

	'''
	Convert timestamps from UTC to local time.
	:param df: dataframe with a site variable and datetime variable in UTC
	:param offset_list: list of timezone offset integers
	:param site_var: name of site variable as a string
	:param time_var: name of datetime variable as a string
	:return: dataframe with datetime converted to local time
	'''

	offset = df[site_var].map(lambda s: np.timedelta64(offset_list[s], 'h'))
	df[time_var] += offset
	return df


def reindex_site_time(df, t_start, t_end,
                      site_var='site_id',
                      time_var='timestamp'):

	'''
	Reindex the dataframe to include a timestamp for every hour within the time
	interval at every site.
	:param df: dataframe with a site variable and datetime variable
	:param t_start: the starting timestamp in the format
					'{month}/{day}/{year} hh:mm:ss'
	:param t_end: the ending timestamp in the same format as t_start
	:param site_var: name of site variable as a string
	:param time_var: name of datetime variable as a string
	:return: dataframe with full timestamps
	'''

	sites = df[site_var].unique()
	frame = df.set_index([site_var, time_var])

	frame = frame.reindex(
		pd.MultiIndex.from_product([
			sites,
			pd.date_range(start=t_start, end=t_end, freq='H')
		])
	)

	frame.index.rename([site_var, time_var], inplace=True)
	return frame.reset_index()


def fill_missing(df,
                 cub_interp_vars=config.cub_interp_vars,
                 lin_interp_vars=config.lin_interp_vars,
                 site_var='site_id'):

	'''
	Fill missing values by site using cubic interpolation, linear
	interpolation, or forward fill. Sites missing 100% of the values will not
	be filled.
	:param df: dataframe with a site variable
	:param cub_interp_vars: list of variable names to use cubic interpolation
	:param lin_interp_vars: list of variable names to use linear interpolation
	:param site_var: name of site variable as a string
	:return: dataframe with missing data filled
	'''

	for col in df.columns:

		if col in cub_interp_vars:
			df[col] = df.groupby(site_var)[col].transform(
				lambda s: s.interpolate('cubic', limit_direction='both')
					.fillna(method='ffill')
					.fillna(method='bfill'))

		elif col in lin_interp_vars:
			df[col] = df.groupby(site_var)[col].transform(
				lambda s: s.interpolate('linear', limit_direction='both')
					.fillna(method='ffill')
					.fillna(method='bfill'))

		else:
			df[col] = df.groupby(site_var)[col].transform(
				lambda s: s.fillna(method='ffill')
					.fillna(method='bfill'))

	return df


def copy_site_data(df, copy_from_site, copy_to_site, var_to_copy,
                   site_var='site_id'):

	'''
	Copy a column from one site to another site. This is used to fill missing
	data in sites with 100% missing values.
	:param df: dataframe with a site variable
	:param copy_from_site: integer of site to copy data from
	:param copy_to_site: integer of site to copy data to
	:param var_to_copy: name of variable to copy
	:param site_var: name of site variable as a string
	:return: dataframe with missing data filled
	'''

	i_from = df[df[site_var] == copy_from_site].index
	i_to = df[df[site_var] == copy_to_site].index

	df.loc[i_to, var_to_copy] = df.loc[i_from, var_to_copy].values
	return df


def merge_data(meter_df, weather_df, building_df,
               on_mb=['building_id'], on_mbw=['site_id', 'timestamp']):

	'''
	Combine the meter, weather, and building data.
	:param meter_df: dataframe with meter data
	:param weather_df: dataframe with weather data
	:param building_df: dataframe with building data
	:param on_mb: list of variable name(s) to merge meter_df and building_df on
	:param on_mbw: list of variable name(s) to merge the resulting dataframe
				   and weather_df on
	:return: dataframe containing meter, weather, and building data
	'''

	mb = pd.merge(meter_df, building_df, on=on_mb, how='left')
	mbw = pd.merge(mb, weather_df, on=on_mbw, how='left')
	return mbw


