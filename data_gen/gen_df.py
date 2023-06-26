import pandas as pd
from paths import *

df = pd.read_parquet(uri_data_path + "Where-Use/WhereUsed.parquet")

print(df.head())