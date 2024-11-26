import pandas as pd
from rich import print
df = pd.read_parquet('https://huggingface.co/datasets/smangrul/ultrachat-10k-chatml/resolve/main/data/test-00000-of-00001.parquet')

df_train = df.sample(frac=0.995, random_state=200)
df_eval = df.drop(df_train.index)

df_train.to_json("ultrachat_chunk_train.jsonl", orient="records", lines=True)
df_eval.to_json("ultrachat_chunk_eval.jsonl", orient="records", lines=True)

print(df_train.iloc[100]['messages'])
