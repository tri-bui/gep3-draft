import pandas as pd

from gb_model.config import config
from gb_model.processing import preprocessing, pipeline


building = pd.read_pickle('datasets/building.pkl')
wpipe = pipeline.wthr_pipe()
fpipe = pipeline.feat_pipe()


def make_prediction(meter_data, weather_data):

	'''
	Make predictions from meter and weather data.

	:param meter_data: (dictionary) raw meter data in JSON format
	:param weather_data: (dictionary) raw weather data in JSON format

	:return: (dictionary) predictions in JSON format
	'''

	meter = pd.read_json(meter_data)
	weather = pd.read_json(weather_data)

	weather = wpipe.fit_transform(weather)
	df = preprocessing.merge_data(meter, weather, building)
	df = fpipe.fit_transform(df)

	pred = pipeline.pred_pipe(df,
	                    'models/rare_enc/rare_enc',
	                    'models/mean_enc/mean_enc',
	                    'models/scaler/scaler'
	                    'models/xgb/xgb')
	response = {'predictions': pred}
	return response