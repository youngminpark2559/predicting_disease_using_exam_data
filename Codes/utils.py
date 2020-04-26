import pandas as pd
import numpy as np
import os

def load_csv(path):
  csv_df=pd.read_csv(path,encoding='utf8',index_col=None)
  # print("csv_df",csv_df)
  return csv_df

def remove_rows_containing_0_or_null(loaded_csv,column_name):
  loaded_csv=loaded_csv[~loaded_csv[column_name].isin([''])]
  loaded_csv=loaded_csv[~loaded_csv[column_name].isin(['0'])]
  return loaded_csv