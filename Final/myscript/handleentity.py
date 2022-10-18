# %%
import csv
import numpy as np
import pandas as pd
df = pd.read_csv("keywords2.csv")

# %%
df1 = pd.DataFrame()
df1["word"] = df["word"]
df1["int"] = df["0"]+1
# %%
df1.to_csv("word2int.tsv", index=None, sep="\t")
# %%
source_embedding = pd.read_table("chinese.txt",
                                 index_col=0,
                                 sep=' ',
                                 header=None,
                                 quoting=csv.QUOTE_NONE,
                                 names=range(301))
# word, vector
source_embedding.index.rename('word', inplace=True)
# %%
for i in df1.itertuples():
    if i.word not in source_embedding.index:
        df1.at[i.Index, "int"] = 0
df1.replace(0, np.nan, inplace=True)
df1.dropna(inplace=True)
df1["int"].astype(np.int64)
df1.to_csv("word2inting.csv", index=None)
# %%
df1 = pd.read_csv("word2inting.csv")
df2=pd.merge(df1,source_embedding,left_on=["word"],right_index=True)
# %%
df2.drop(["word"],inplace=True,axis=1)
df2.to_csv("entity_embedding.vec",index=None,sep="\t")

# %%
