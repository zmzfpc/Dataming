# %%
import pandas as pd
df=pd.read_csv("news_parsed.tsv",sep="\t")
print(df["id"].value_counts())
# %%
