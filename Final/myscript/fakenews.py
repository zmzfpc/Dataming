# %%
from tqdm import tqdm
import csv
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
import numpy as np
# %%
df = pd.read_csv('同洲媒资库元数据_sample.csv', index_col=0)

res = pd.DataFrame(
    index=df.index,
    columns=['news_id', 'title', 'abstract', 'body', 'url', 'category_label']
)

df['ASSET_TYPE'] = df['ASSET_TYPE'].fillna('')

asset_types = df['ASSET_TYPE'].unique()
asset_types = list(map(lambda a: list(
    filter(lambda x: x != '', a.strip('/').split('/'))), asset_types))
all_asset_types = set(sum(asset_types, []))
print(len(all_asset_types))

res['news_id'] = df['ASSET_ID']
res['title'] = df['TITLE_FULL']
res['abstract'] = df['SUMMAR_MEDIUM']
res['body'] = df['TITLE_BRIEF']
res['url'] = df['FOLDER_ASSET_ID']

# %%
l = []
l1 = []
l2 = []
count = 0
en = pd.read_csv("word2inting.csv")
for i in res.itertuples():
    ll1 = []
    ll2 = []
    count += 1

    for j in en.itertuples():
        d1 = {}
        d2 = {}
        if j.word in i.title:
            d1["Label"] = j.word
            d1["Type"] = "D"
            d1["WikidataId"] = j.int
            d1["Confidence"] = "1"
            d1["OccurrenceOffsets"] = [len(j.word)*20+5]
            d1["SurfaceForms"] = ["{}".format(j.word)]
        if j.word in i.abstract:
            d2["Label"] = j.word
            d2["Type"] = "D"
            d2["WikidataId"] = j.int
            d2["Confidence"] = "1"
            d2["OccurrenceOffsets"] = [len(j.word)*20+5]
            d2["SurfaceForms"] = ["{}".format(j.word)]
        if d1!={}:
            ll1.append(json.dumps(d1, ensure_ascii=False, separators=(',', ':')))
        if d2!={}:
            ll2.append(json.dumps(d2, ensure_ascii=False, separators=(',', ':')))

    l1.append(ll1)
    l2.append(ll2)
    if count % 1000 == 0:
        print(count)
        print(ll1)
        print(l1)
# %%

res['category_label'] = l1
res['category_label1'] = l2
res.drop_duplicates(subset='news_id', keep='first', inplace=True)
res.to_csv('news.tsv', sep='\t', index=None)

# %%
