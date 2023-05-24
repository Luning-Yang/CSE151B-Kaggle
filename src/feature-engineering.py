import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.impute import SimpleImputer

# Over every single 
def polyline_to_trip_duration(polyline):
    return max(polyline.count("[") - 2, 0) * 15

def parse_time(x):
  # We are using python's builtin datetime library
  # https://docs.python.org/3/library/datetime.html#datetime.date.fromtimestamp

  # Each x is essentially a 1 row, 1 column pandas Series
    dt = datetime.fromtimestamp(x["TIMESTAMP"])
    return dt.year, dt.month, dt.day, dt.hour, dt.weekday()



if __name__ == "__main__":
	df_tr = pd.read_csv("../data/train.csv")
	# This code creates a new column, "LEN", in our dataframe. The value is
	# the (polyline_length - 1) * 15, where polyline_length = count("[") - 1
	df_tr["LEN"] = df_tr["POLYLINE"].apply(polyline_to_trip_duration)
	df_tr[["YR", "MON", "DAY", "HR", "WK"]] = df_tr[["TIMESTAMP"]].apply(parse_time, axis=1, result_type="expand")

	# One-hot encoding CALL_TYPE
	df_encoded_call_type = pd.get_dummies(df_tr['CALL_TYPE'], prefix='call_type')
	# One-hot encoding TAXI_ID
	df_encoded_taxi_id = pd.get_dummies(df_tr['TAXI_ID'], prefix='taxi_id')

	# impute and One-hot encoding ORIGIN_STAND
	imputer = SimpleImputer(strategy='most_frequent')
	df_tr['ORIGIN_STAND'] = imputer.fit_transform(df_tr[['ORIGIN_STAND']])
	df_encoded_origin_stand = pd.get_dummies(df_tr['ORIGIN_STAND'], prefix='origin_stand')


	# Join the encoded dataframes back with the original dataframe
	df_tr = pd.concat([df_tr, df_encoded_call_type, df_encoded_taxi_id, df_encoded_origin_stand], axis=1)
	df_tr = df_tr.drop(['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 
	                    'ORIGIN_STAND', 'TAXI_ID', 'TIMESTAMP', 
	                    'DAY_TYPE', 'MISSING_DATA', 'POLYLINE'], axis=1)
	df_tr.to_csv('../data/processed_train.csv',index=False)


