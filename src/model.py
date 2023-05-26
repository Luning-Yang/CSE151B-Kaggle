import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
# import lightgbm as lgb


# Over every single 
def polyline_to_trip_duration(polyline):
    return max(polyline.count("[") - 2, 0) * 15

def parse_time(X):
    dt = X['TIMESTAMP'].apply(datetime.fromtimestamp)
    X['YR'] = dt.dt.year
    X['MON'] = dt.dt.month
    X['DAY'] = dt.dt.day
    X['HR'] = dt.dt.hour
    X['WK'] = dt.dt.weekday
    return X

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)


# Define the columns to include in the ColumnTransformer
columns_to_include = ['CALL_TYPE', 'TAXI_ID', 'ORIGIN_STAND']

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('impute', SimpleImputer(strategy='most_frequent'), ['ORIGIN_STAND']),
        ('ohe', OneHotEncoder(), columns_to_include),
    ])



pipeline = Pipeline(steps=[
    ('column_dropper', ColumnDropper(columns=['TRIP_ID', 'ORIGIN_CALL', 'DAY_TYPE', 'MISSING_DATA'])),
    ('time_transformer', FunctionTransformer(parse_time, validate=False)),
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())  # enable GPU usage
])

#...
# the rest of the code remains the same

if __name__ == "__main__":
	# Load the data
	df_tr = pd.read_csv("../data/train.csv")
	X = df_tr.drop('POLYLINE', axis=1)
	y = df_tr['POLYLINE'].apply(polyline_to_trip_duration)
	pipeline.fit(X, y)

	# Load the test set
	df_te = pd.read_csv("../data/test_public.csv")
	# Preprocess and predict using the trained pipeline
	predictions = pipeline.predict(df_te)
	df_submission = pd.DataFrame()
	df_submission['TRIP_ID'] = df_te['TRIP_ID']
	df_submission['TRAVEL_TIME'] = predictions
	df_submission.to_csv("my_pred.csv", index=None)



	# Fit and transform the data
	# X_transformed = preprocessor.fit_transform(X)
	# df_tr.to_csv('../data/processed_train.csv',index=False)


