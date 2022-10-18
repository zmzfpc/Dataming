# %%
import pandas as pd
df=pd.read_csv("behaviors.tsv",sep="\t")

# %%
df1=df.iloc[0:75060]
df2=df.iloc[75060:102000]
df3=df.iloc[102000:125100]

# %%

df1.to_csv("behaviors1.tsv",index=None,sep="\t")
df2.to_csv("behaviors2.tsv",index=None,sep="\t")
df3.to_csv("behaviors3.tsv",index=None,sep="\t")
# %%
