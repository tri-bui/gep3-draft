import config
import preprocessing as pp
import features as ft
import prediction as pn

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


# Weather data preprocessing

wthr_pipe = Pipeline(
	[
		('time_converter', pp.TimeConverter(timezones=config.tz_offsets)),
		('time_reindexer', pp.TimeReindexer()),
		('missing_imputer',
		 pp.MissingImputer(cub_vars=config.cub_interp_vars,
		                   lin_vars=config.lin_interp_vars)),
		('data_copier', pp.DataCopier())
	]
)

# wpipe = wthr_pipe()
# df = wpipe.fit_transform(df)

# b = pd.read_csv('datasets/building.csv')
# df = pp.merge_data(m, w, b)

feat_pipe = Pipeline(
	[
		('weather_extractor', ft.WeatherExtractor()),
		('time_extractor', ft.TimeExtractor()),
		('holiday_extractor', ft.HolidayExtractor(countries=config.countries)),
		('feat_selector', ft.FeatSelector(feats=config.feats))
	]
)

# fpipe = feat_pipe()
# df = fpipe.fit_transform(df)

# # Rare label categorical encoders
# rare_enc0 = joblib.load('transformers/rare_enc/rare_enc0.pkl')
# rare_enc1 = joblib.load('transformers/rare_enc/rare_enc1.pkl')
# rare_enc2 = joblib.load('transformers/rare_enc/rare_enc2.pkl')
# rare_enc3 = joblib.load('transformers/rare_enc/rare_enc3.pkl')
# re = [rare_enc0, rare_enc1, rare_enc2, rare_enc3]

# # Mean categorical encoders
# mean_enc0 = joblib.load('transformers/mean_enc/mean_enc0.pkl')
# mean_enc1 = joblib.load('transformers/mean_enc/mean_enc1.pkl')
# mean_enc2 = joblib.load('transformers/mean_enc/mean_enc2.pkl')
# mean_enc3 = joblib.load('transformers/mean_enc/mean_enc3.pkl')
# me = [mean_enc0, mean_enc1, mean_enc2, mean_enc3]

# # Standard scalers
# scaler0 = joblib.load('transformers/scaler/scaler0.pkl')
# scaler1 = joblib.load('transformers/scaler/scaler1.pkl')
# scaler2 = joblib.load('transformers/scaler/scaler2.pkl')
# scaler3 = joblib.load('transformers/scaler/scaler3.pkl')
# ss = [scaler0, scaler1, scaler2, scaler3]

# LGB
# pred = pn.pred_pipe(df, re, me, ss, 'models/lgb/lgb', use_xgb=False)

# XGB
# pred = pn.pred_pipe(df, re, me, ss, 'models/xgb/xgb')