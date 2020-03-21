import config
import features
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# # Rare label categorical encoders
# rare_enc0 = joblib.load('transformers/rare_enc/rare_enc0.pkl')
# rare_enc1 = joblib.load('transformers/rare_enc/rare_enc1.pkl')
# rare_enc2 = joblib.load('transformers/rare_enc/rare_enc2.pkl')
# rare_enc3 = joblib.load('transformers/rare_enc/rare_enc3.pkl')
# rare_enc = [rare_enc0, rare_enc1, rare_enc2, rare_enc3]
#
# # Mean categorical encoders
# mean_enc0 = joblib.load('transformers/mean_enc/mean_enc0.pkl')
# mean_enc1 = joblib.load('transformers/mean_enc/mean_enc1.pkl')
# mean_enc2 = joblib.load('transformers/mean_enc/mean_enc2.pkl')
# mean_enc3 = joblib.load('transformers/mean_enc/mean_enc3.pkl')
# mean_enc = [mean_enc0, mean_enc1, mean_enc2, mean_enc3]
#
# # Standard scalers
# scaler0 = joblib.load('transformers/scaler/scaler0.pkl')
# scaler1 = joblib.load('transformers/scaler/scaler1.pkl')
# scaler2 = joblib.load('transformers/scaler/scaler2.pkl')
# scaler3 = joblib.load('transformers/scaler/scaler3.pkl')
# scaler = [scaler0, scaler1, scaler2, scaler3]
#
# # LightGBM models
# lgb0 = joblib.load('models/lgb/lgb0.pkl')
# lgb1 = joblib.load('models/lgb/lgb1.pkl')
# lgb2 = joblib.load('models/lgb/lgb2.pkl')
# lgb3 = joblib.load('models/lgb/lgb3.pkl')
# lgb = [lgb0, lgb1, lgb2, lgb3]
#
# # XGBoost Models
# xgb0 = joblib.load('models/xgb/xgb0.pkl')
# xgb1 = joblib.load('models/xgb/xgb1.pkl')
# xgb2 = joblib.load('models/xgb/xgb2.pkl')
# xgb3 = joblib.load('models/xgb/xgb3.pkl')
# xgb = [xgb0, xgb1, xgb2, xgb3]


def split(df,
          meter_var='meter'):

	'''
	Split data by meter type.

	:param df: (Pandas dataframe) full data
	:param meter_var: (string) name of meter type variable

	:return: list of dataframes split by meter type
	'''

	dfs = []
	for m in range(4):
		dfm = df[df[meter_var] == m].drop(meter_var, axis=1)
		dfs.append(dfm)
	return dfs


def transform(df, rare_enc, mean_enc, scaler):

	'''
	Transform data using rare label categorical encoding, mean
	categorical encoding, and standard scaling. Features will be
	selected on this resulting data.

	:param df: (Pandas dataframe) data with variables to be selected
	:param rare_enc: (Feature-engine categorical encoder object)
					 fitted rare label categorical encoder
	:param mean_enc: (Feature-engine categorical encoder object)
					 fitted mean categorical encoder
	:param scaler: (Scikit-learn preprocessing object)
				   fitted standard scaler

	:return: transformed dataframe with selected features
	'''

	encoded = rare_enc.transform(df)
	encoded = mean_enc.transform(encoded)
	scaled = scaler.transform(encoded)
	scaled = pd.DataFrame(scaled, columns=df.columns)
	df = features.select_feats(scaled)
	return df


def predict(df, model_path, xgb=True):

	'''
	Make predictions using a trained model.

	:param df: (Pandas dataframe) data with features matching
			   training data
	:param model_path: (string) path to trained model
	:param xgb: (boolean) whether or not to predict using a XGBoost model

	:return: non-negative predictions
	'''

	if xgb:
		df = xgb.DMatrix(df)

	model = joblib.load(model_path)
	pred = model.predict(df)
	pred[pred < 0] = 0
	return pred


def inverse_transform(df, sqft_var='square_feet', target_var='meter_reading'):

	'''
	Inverse transform predictions. The target variable in the training data
	was standardized using the square_feet variable and then log-transformed.

	:param df: (Pandas dataframe) data with square footage and target variables
	:param sqft_var: (String) name of square footage variable
	:param target_var: (String) name of target variable

	:return: inverse-transformed dataframe
	'''

	df[target_var] = np.expm1(df[target_var])
	df[target_var] *= df[sqft_var] / df[sqft_var].mean()
	return df


def convert_site0_units(df,
                        site_var='site_id',
                        meter_var='meter',
                        target_var='meter_reading'):

	'''
	Convert site 0 meter 0 readings from kWh back to kBTU and site 0 meter 1
	readings from tons back to kBTU. Site 0 meter readings in the training data
	were recorded in kBTU, but the model was trained on units kWh and tons for
	meter 0 and meter 1 respectively.

	:param df: (Pandas dataframe) data with site, meter, and target variables
	:param site_var: (String) name of site variable
	:param meter_var: (String) name of meter type variable
	:param target_var: (String) name of target variable

	:return: dataframe with units in site 0 converted
	'''

	df.loc[(df[site_var] == 0) & (df[meter_var] == 0), target_var] *= 3.4118
	df.loc[(df[site_var] == 0) & (df[meter_var] == 1), target_var] *= 12
	return df


def pred_lgb(df, rare_enc_list, mean_enc_list, sclr_list, model_path,
             output_path='predictions/pred.csv',
             sqft_var='square_feet',
             target_var='meter_reading'):

	'''
	Make predictions using LightGBM.

	:param df: (Pandas dataframe) preprocessed data with listed variables
	:param rare_enc_list: (List of Feature-engine categorical encoder objects)
						  trained rare label categorical encoders
	:param mean_enc_list: (List of Feature-engine categorical encoder objects)
						  trained mean categorical encoders
	:param sclr_list: (List of Scikit-learn preprocessing objects)
					  trained standard scalers
	:param model_path: (String) path to trained LightGBM model
	:param output_path: (String) path to save predictions
	:param sqft_var: (String) name of square footage variable
	:param target_var: (String) name of target variable

	:return: predictions in a dataframe (Kaggle submission format)
	'''

	df.reset_index(inplace=True)
	df_list = split(df)
	preds = []

	for i in range(4):
		X = transform(df_list[i], rare_enc_list[i], mean_enc_list[i], sclr_list[i])
		y_pred = predict(X, model_path + str(i) + '.pkl', xgb=False)
		y = df_list[i][[sqft_var]].copy()
		y[target_var] = y_pred
		y = inverse_transform(y)
		preds.append(y)

	pred = pd.concat(preds).sort_index().reset_index()
	pred = pd.merge(df[['index', 'site_id', 'meter']], pred, on='index', how='left')
	pred = convert_site0_units(pred)
	pred = pred[['index', 'meter_reading']]
	pred.columns = ['row_id', 'meter_reading']
	pred.to_csv(output_path)
	return pred


def pred_xgb(df, rare_enc_list, mean_enc_list, sclr_list, model_path,
             output_path='predictions/pred.csv',
             sqft_var='square_feet',
             target_var='meter_reading'):

	'''
	Make predictions using XGBoost.

	:param df: (Pandas dataframe) preprocessed data with listed variables
	:param rare_enc_list: (List of Feature-engine categorical encoder objects)
						  trained rare label categorical encoders
	:param mean_enc_list: (List of Feature-engine categorical encoder objects)
						  trained mean categorical encoders
	:param sclr_list: (List of Scikit-learn preprocessing objects)
					  trained standard scalers
	:param model_path: (String) path to trained XGBoost model
	:param output_path: (String) path to save predictions
	:param sqft_var: (String) name of square footage variable
	:param target_var: (String) name of target variable

	:return: predictions in a dataframe (Kaggle submission format)
	'''

	df.reset_index(inplace=True)
	df_list = split(df)
	preds = []

	for i in range(4):
		X = transform(df_list[i], rare_enc_list[i], mean_enc_list[i], sclr_list[i])
		y_pred = predict(X, model_path + str(i) + '.pkl')
		y = df_list[i][[sqft_var]].copy()
		y[target_var] = y_pred
		y = inverse_transform(y)
		preds.append(y)

	pred = pd.concat(preds).sort_index().reset_index()
	pred = pd.merge(df[['index', 'site_id', 'meter']], pred, on='index', how='left')
	pred = convert_site0_units(pred)
	pred = pred[['index', 'meter_reading']]
	pred.columns = ['row_id', 'meter_reading']
	pred.to_csv(output_path)
	return pred