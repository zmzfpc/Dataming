# %%
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
# %%

df = pd.read_csv('播放记录.csv', index_col=0)

res = pd.DataFrame()

res['user_id'] = df['v']
res['time'] = df['m']
res['resource_id'] = df['d']

res['duration'] = df['f']
res['watchtime'] = df['u']
res['isclicked'] = ((res['watchtime'] / res['duration'])
                    > 0.25).astype('int').astype("str")
print(res['isclicked'].value_counts())
res.sort_values('time', inplace=True)

res.to_csv('fakebehavior.tsv', sep='\t', index=None)

# %%
out = pd.DataFrame()
out["user_id"] = res['user_id']
out['time'] = res['time']
out["chick_list"] = ""
out["clicked"] = res['resource_id']+"-"+res['isclicked']

out = out.reset_index(drop=True)

out.to_csv('fakebehavior1.tsv', sep='\t')
# %%
l = {}
ll = []
for i in range(len(out)):
    if i % 1000 == 0:
        print(i)
    if out.iloc[i]["user_id"] in l:
        out.iloc[i]["chick_list"] = l[out.iloc[i]["user_id"]]
        ll.append(l[out.iloc[i]["user_id"]])
    else:
        ll.append("")
    if out.iloc[i]["clicked"].split("-")[1] == "1":
        if out.iloc[i]["user_id"] in l:
            l[out.iloc[i]["user_id"]
              ] += out.iloc[i]["clicked"].split("-")[0]+" "
        else:
            l[out.iloc[i]["user_id"]] = out.iloc[i]["clicked"].split(
                "-")[0]+" "
out.to_csv('fakebehavior2.tsv', sep='\t')
# %%
ll1 = np.array(ll)
out1 = pd.DataFrame(ll1, columns=["1"])

# %%

out1.to_csv('fakebehavior2.csv', index=None)
# %%
out["chick_list"] = out1["1"]
# %%
out.to_csv('fakebehavior3.csv', index=None)

# %%
out = pd.read_csv('fakebehavior3.csv')
out.replace("", np.nan, inplace=True)
out.dropna(inplace=True)
out = out.reset_index(drop=True)
out.to_csv('fakebehavior.tsv', sep='\t')


