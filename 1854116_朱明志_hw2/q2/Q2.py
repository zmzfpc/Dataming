# %%
# 引入包
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# %%
input1 = pd.read_csv("data/CollectionRecord.csv",encoding="gbk",)
input2 = pd.read_csv("data/ChannelRecordSample.csv",encoding="gbk")
print(input1.shape)
print(input2.shape)
# %%
# 删除冗余字段
input1.drop(["ID","TYPE", "URL","ITEMCODE","MD5","STATUS"], axis=1,inplace=True)
input2.drop(["PT_TIME", "RN"], axis=1,inplace=True)
input1.drop(["NAME","TIME"], axis=1,inplace=True)
input2.drop(["L_CHANNEL_NAME"], axis=1,inplace=True)
print(input1.shape)
print(input2.shape)
for i in input1:
    print(input1[i].value_counts())
print("----------")
for i in input2:
    print(input2[i].value_counts())
input1.to_csv("csvoutput/output1.csv", index=None, float_format='%.0f')
input2.to_csv("csvoutput/output2.csv", index=None, float_format='%.0f')
print("csv save!")
# %%
input1 = pd.read_csv("csvoutput/output1.csv",encoding="gbk",)
input2 = pd.read_csv("csvoutput/output2.csv",encoding="gbk")

# %%
input1.drop(["PORTAL_VER"], axis=1,inplace=True)
#input1.drop([""], axis=1,inplace=True)
input1["STBID"]=input1["STBID"].apply(
    lambda x: np.nan if type(x)==type("xxssd") else x
    )
mid1=input1.drop(["CODE","FOLDERCODE"],axis=1,inplace=False)
print(mid1.shape)
mid1=mid1.dropna()

# %%
input2.drop(["OPK"], axis=1,inplace=True)
input2.sort_values(by="CNT", inplace=True, ascending=False)
#input2.drop_duplicates(subset="STBID", inplace=True, keep="first")
#input2.drop(["STBID"], axis=1,inplace=True)
input2.to_csv("csvoutput/output4.csv", index=None, float_format='%.3f')

for i in input1:
    print(input1[i].value_counts())
print("----------")
for i in input2:
    print(input2[i].value_counts())

# %%

me=pd.merge(input2,mid1,on="STBID",how="inner")
me.info()
# %%
me.drop(["STBID"], axis=1,inplace=True)
me.to_csv("csvoutput/output5.csv", index=None, float_format='%.3f')
# %%
sam = me.sample(n=1000, random_state=110, axis=0)
# %%
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
input2_std = std.fit_transform(sam)
#input2_std=mid1
#print(input2_std)
# %%
df_number = mid1.select_dtypes(exclude=['object']).reset_index(drop=True)
# df_number_normalized = (df_number - df_number.mean()) / df_number.std()
mid_normalized = (df_number - df_number.min()) / (df_number.max() - df_number.min())
pca = PCA(n_components=2)
out = pca.fit(mid_normalized)
out = out.transform(mid_normalized)
plt.scatter(out[:, 0], out[:, 1])
plt.show()
# %%
plt.matshow(me.corr(method="pearson"))
plt.colorbar()
plt.show()
# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
# k-means聚类遍历
l=[]
for i in range(2,12):
    print(i)
    km=KMeans(n_clusters=i)
    clf1 = km.fit_predict(input2_std)
    s = silhouette_score(input2_std, clf1)
    print("The silhouette_score= {}".format(s))
    l.append([i,s])
    plt.scatter(i, s)
    #n_s_bigger_than_zero = (silhouette_samples(out, clf1) > 0).sum()
    #print("{}/{}\n".format(n_s_bigger_than_zero, out.shape[0]))
l=np.array(l)
plt.plot(l[:,0],l[:,1])
plt.show()
# %%
# 层次聚类
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.metrics import silhouette_score, silhouette_samples
plt.figure(figsize=(20,6))
Z = linkage(input2_std, method='ward', metric='euclidean')
p = dendrogram(Z, 0)
df=pd.DataFrame(Z)
df.to_csv("csvoutput/out.csv")
plt.show()

# %%
import scipy.cluster.hierarchy as sch
df=pd.read_csv("csvoutput/out.csv")
l=[df["0"],df["1"],df["2"],df["3"]]
l=np.array(l)
l=l.T
clf2= sch.fcluster(l,15,"distance") 
s = silhouette_score(input2_std, clf2)
print("The silhouette_score= {}".format(s))
# %%
l=[]
l.append(clf1)
l.append(clf2)
l=np.array(l)
l=l.T
print(l.shape)
df=pd.DataFrame(l,columns=["kmeans_label"]+["hierarchical_label"])
df.to_csv("csvoutput/result.csv")
# %%
plt.scatter(out[:, 0], out[:, 1],c=clf2)
ex = 0.5
step = 0.01
xx, yy = np.meshgrid(np.arange(out[:, 0].min() - ex, out[:, 0].max() + ex, step),
                        np.arange(out[:, 1].min() - ex, out[:, 1].max() + ex, step)) 
zz = km.predict(np.c_[xx.ravel(), yy.ravel()])
zz.shape = xx.shape
plt.contourf(xx, yy, zz, alpha= 0.1)
plt.show()
s = silhouette_score(out, clf2)
print("The silhouette_score= {}".format(s))
# %%
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, silhouette_samples
# k-means聚类
km=KMeans(n_clusters=5)

clf1 = km.fit_predict(out)
plt.scatter(out[:, 0], out[:, 1],c=clf1)
ex = 0.5
step = 0.01
xx, yy = np.meshgrid(np.arange(out[:, 0].min() - ex, out[:, 0].max() + ex, step),
                        np.arange(out[:, 1].min() - ex, out[:, 1].max() + ex, step)) 
zz = km.predict(np.c_[xx.ravel(), yy.ravel()])
zz.shape = xx.shape
plt.contourf(xx, yy, zz, alpha= 0.1)
plt.show()
s = silhouette_score(out, clf1)
print("The silhouette_score= {}".format(s))
# %%
# 聚类结果分析可视化
#mid=pd.read_csv("csvoutput/output3.csv")

#mid = (df_number - df_number.min()) / (df_number.max() - df_number.min())
sam["label"]=clf1
#mid.sort_values(by="label", inplace=True, ascending=False)
grouped=sam.groupby(sam["label"])
print(grouped.mean())
x=range(1) 
label_list=[]
mea=grouped.mean()
count=0
for i in mea:
    count=count+1
    if count>=2:
        label_list.append(i)
    if count==2:
        break
rects1 = plt.bar(x, height=mea[label_list].iloc[0], width=0.2, alpha=0.8, color='red', label="一")
rects2 = plt.bar([i + 0.2 for i in x], height=mea[label_list].iloc[1], width=0.2, color='yellow', label="二")
rects3 = plt.bar([i + 0.4 for i in x], height=mea[label_list].iloc[2], width=0.2, color='blue', label="二")
rects2 = plt.bar([i + 0.6 for i in x], height=mea[label_list].iloc[3], width=0.2, color='green', label="二")
rects3 = plt.bar([i + 0.8 for i in x], height=mea[label_list].iloc[4], width=0.2, color='black', label="二")

plt.xticks([index + 0.4 for index in x], label_list)
plt.xlabel("Attributes")
plt.title("Cluster feature means")
plt.show()
# %%
