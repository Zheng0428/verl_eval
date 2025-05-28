import pandas as pd

# Specify the path to your Parquet file
parquet_file_path = '/mnt/hdfs/tiktok_aiic/user/liuqian/rl_datasets/deepscaler/aime.parquet'

# Read the Parquet file into a DataFrame
df = pd.read_parquet(parquet_file_path, engine='pyarrow')  # or engine='fastparquet'

# Display the DataFrame
print(df.head()["prompt"][0])