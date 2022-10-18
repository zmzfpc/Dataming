# %%
# 引入包
import numpy as np
from numpy.lib.shape_base import split
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics import mutual_info_score
# 导入数据
input = pd.read_csv("data/log.csv")
npy = np.array([input["time"]])
npy = npy.T
out = list()
# %%
# 原始数据读取
count=0
for row in npy:
    row = str(row).split("'")[1]
    row1 = row.split(":")[0]
    row2 = row.split(":")[1]
    row3 = row.split(":")[2]
    if row1 == "":
        row1 = np.nan
    if row2 == "":
        row2 = np.nan
    if row3 == "":
        row3 = np.nan
    out.append([row1, row2, row3])
    count=count+1

out = np.array(out)
# %%
# 导入一个pandas数据框架下
df = pd.DataFrame(out, columns=['h'] + ['m'] + ['s'])
df = df.replace("nan", np.nan)
df.to_csv("test.csv", index=None, float_format='%.3f')
print(df)
print(df.shape)
# %%
# 数据缺失值处理部分
# 特殊值填充
dorp = df.fillna(value='nan')
dorp.to_csv("test1.csv", index=None, float_format='%.3f')
# 元组删除
dorp = df.dropna()
dorp.to_csv("test2.csv", index=None, float_format='%.3f')
# k-mean填充
dorp = df.fillna(axis=0, method='ffill')
dorp.to_csv("test3.csv", index=None, float_format='%.3f')
# 平均值填充
df1 = pd.DataFrame(df.dropna(), dtype=np.int32)
dorp = df.fillna(value={'h': int(df1['h'].mean()), 'm': int(
    df1['m'].mean()), 's': int(df1['s'].mean())})
dorp.to_csv("test4.csv", index=None, float_format='%.3f')
print("csv save")
# %%
# 数据采样部分
# 随机采样
df=df.fillna(axis=0, method='ffill')
sam = df.sample(n=int(len(df)/6), random_state=110, axis=0)
sam.to_csv("sample1.csv", index=None, float_format='%.3f')

# 分层抽样 分组
frame=list()

for i in range(int(len(df)/6)):
    frame.append(df.iloc[i*6:i*6+6].sample(n=1,random_state=(i+1)*10))
sam=pd.concat(frame)
print(sam)
#    返回值：抽样后的数据框
sam.to_csv("sample2.csv", index=None, float_format='%.3f')

# 等距采样
frame=list()

for i in range(int(len(df)/6)):
    frame.append(df.iloc[i*6+1:i*6+2])
sam=pd.concat(frame)
print(sam)
#    返回值：抽样后的数据框
sam.to_csv("sample3.csv", index=None, float_format='%.3f')

print("csv save")

# %%
