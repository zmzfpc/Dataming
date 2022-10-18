# %%
# 引入包
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

import pandas as pd

# %%
# 导入数据
input = pd.read_csv("data/diabetic_data.csv")

# %%
# 缺失值处理
input = input.replace("?", np.nan)
print(input.shape)
print(input.isnull().sum())
for i in input:
    if input.isnull().sum()[i] > 40000:
        input.drop(i, axis=1, inplace=True)

print(input.isnull().sum())
print(len(input.iloc[1]))

# %%
# NZV方差近0列删除
for i in input:
    if input[i].value_counts().iloc[0] > 90000:
        # print(input[i].value_counts().iloc[0])
        input.drop(i, axis=1, inplace=True)
        print(i)
print(input.isnull().sum())
print(len(input.iloc[1]))

# %%
# 多次见面即重复行删除仅保留住院时间最长行
input.sort_values(by="time_in_hospital", inplace=True, ascending=False)
input.drop_duplicates(subset="patient_nbr", inplace=True, keep="first")
# for i in input:
print(input["patient_nbr"].value_counts())
print(len(input))
input.to_csv("csvoutput/output1.csv", index=None, float_format='%.3f')

# %%
# 删除两列id号并删除缺失值数据行
mid = input.drop(["patient_nbr", "encounter_id"], axis=1)
#mid=pd.read_csv("csvoutput/output1.csv")
mid = mid.replace("Unknown/Invalid", np.nan)
mid = mid.dropna()
print(mid.shape)
mid.to_csv("csvoutput/output2.csv", index=None, float_format='%.0f')
# %%
mid = pd.read_csv("csvoutput/output2.csv")


# %%
# 对字符串分类值全部处理为数字分类值

for i in mid:
    print(mid[i].value_counts())
mid = mid.replace("Caucasian", 0)
mid = mid.replace("Hispanic", 1)
mid = mid.replace("AfricanAmerican", -1)
mid = mid.replace("Other", 2)
mid = mid.replace("Asian", -2)
mid = mid.replace("Female", -1)
mid = mid.replace("Male", 1)
mid = mid.replace("Unknown/Invalid", np.nan)
# print(mid["age"].value_counts())
mid = mid.replace("[0-10)", -2)
mid = mid.replace("[10-20)", -2)
mid = mid.replace("[20-30)", -2)
mid = mid.replace("[30-40)", -2)
mid = mid.replace("[40-50)", -2)
mid = mid.replace("[50-60)", -1)
mid = mid.replace("[60-70)", 0)
mid = mid.replace("[70-80)", 1)
mid = mid.replace("[80-90)", 2)
mid = mid.replace("[90-100)", 2)

mid = mid.dropna()

# %%
def infl(x): return int(float(x))


mid["diag_1"] = mid["diag_1"].apply(
    lambda x: 1 if ("V" in str(x) or "E" in str(x)) else
    (2 if infl(x) in range(390, 460) or infl(x) == 785 else
     (3 if infl(x) in range(460, 520) or infl(x) == 786 else
      (4 if infl(x) in range(520, 580) or infl(x) == 787 else
       (5 if infl(x) == 250 else
        (6 if infl(x) in range(800, 1000) else
         (7 if infl(x) in range(710, 740) else
          (8 if infl(x) in range(580, 630) or infl(x) == 788 else
           (9 if infl(x) in range(140, 240) else
            (10 if infl(x) in range(630, 680) else 1))))))))))
mid["diag_2"] = mid["diag_2"].apply(
    lambda x: 1 if ("V" in str(x) or "E" in str(x)) else
    (2 if infl(x) in range(390, 460) or infl(x) == 785 else
     (3 if infl(x) in range(460, 520) or infl(x) == 786 else
      (4 if infl(x) in range(520, 580) or infl(x) == 787 else
       (5 if infl(x) == 250 else
        (6 if infl(x) in range(800, 1000) else
         (7 if infl(x) in range(710, 740) else
          (8 if infl(x) in range(580, 630) or infl(x) == 788 else
           (9 if infl(x) in range(140, 240) else
            (10 if infl(x) in range(630, 680) else 1))))))))))
mid["diag_3"] = mid["diag_3"].apply(
    lambda x: 1 if ("V" in str(x) or "E" in str(x)) else
    (2 if infl(x) in range(390, 460) or infl(x) == 785 else
     (3 if infl(x) in range(460, 520) or infl(x) == 786 else
      (4 if infl(x) in range(520, 580) or infl(x) == 787 else
       (5 if infl(x) == 250 else
        (6 if infl(x) in range(800, 1000) else
         (7 if infl(x) in range(710, 740) else
          (8 if infl(x) in range(580, 630) or infl(x) == 788 else
           (9 if infl(x) in range(140, 240) else
            (10 if infl(x) in range(630, 680) else 1))))))))))
mid["admission_source_id"] = mid["admission_source_id"].apply(
    lambda x: 1 if x in [1,2,3] else
    (2 if x in [4,5,6,10,18,19,22,25,26] else
     (3 if x in [12,13,14,23,24] else
      (4 if x in [9,15,17,20,21] else 5 
          ))))
print(mid["diag_1"].value_counts())
print(mid["diag_2"].value_counts())
print(mid["diag_3"].value_counts())
# %%
mid.replace("None",0,inplace=True)
mid.replace(">8",1,inplace=True)
mid.replace(">7",2,inplace=True)
mid.replace("Norm",-1,inplace=True)
mid.replace("No",0,inplace=True)
mid.replace("Up",-1,inplace=True)
mid.replace("Down",-2,inplace=True)
mid.replace("Steady",1,inplace=True)
mid.replace("Ch",1,inplace=True)
mid.replace("Yes",1,inplace=True)
print(mid["A1Cresult"].value_counts())
print(mid["metformin"].value_counts())
print(mid["glipizide"].value_counts())
print(mid["insulin"].value_counts())
print(mid["change"].value_counts())
print(mid["diabetesMed"].value_counts())

# %%
mid["discharge_disposition_id"] = mid["discharge_disposition_id"].apply(
    lambda x: 1 if x in [1,2,3] else
    (2 if x in [4,5,6,10,18,19,22,25,26] else
     (3 if x in [12,13,14,23,24] else
      (4 if x in [9,15,17,20,21] else 5 
          ))))
# %%
tag=mid["readmitted"]
mid.drop("readmitted", axis=1, inplace=True)
mid.to_csv("csvoutput/output3.csv", index=None, float_format='%.0f')
# %%
# 输出转化的数据
mid.replace("NO", 0, inplace=True)
mid.replace("<30", -1, inplace=True)
mid.replace(">30", 1, inplace=True)
for i in mid:
    print(mid[i].value_counts())
#mid = mid.drop(["patient_nbr", "encounter_id"], axis=1)
mid.to_csv("csvoutput/output3.csv", index=None, float_format='%.0f')
# %%
mid=pd.read_csv("csvoutput/output3.csv")
# %%
plt.matshow(mid.corr(method="pearson"))
plt.colorbar()
plt.show()
mmid=mid.corr(method="pearson")
print(mid.shape)
cc=0
for i in mmid:
    for j in mmid[i]:
        if j>0.8 and j!=1:
            print(i)
            cc=cc+1
print("相关系数大于0.8的对数:"+str(int(cc/2)))     
# %%
for i in mid:
    print(i)
mmid=pd.DataFrame()
#mmid["time_in_hospital"]=mid["time_in_hospital"]   
mmid["num_lab_procedures"]=mid["num_lab_procedures"]   
mmid["num_procedures"]=mid["num_procedures"]   
mmid["num_medications"]=mid["num_medications"]   
mmid["number_outpatient"]=mid["number_outpatient"]   
#mmid["number_inpatient"]=mid["number_inpatient"]   
#mmid["number_diagnoses"]=mid["number_diagnoses"]   
#mmid["admission_type_id"]=mid["admission_type_id"]   
#mmid["diag"]=mid["diag_2"]+mid["diag_3"]+mid["diag_1"]
print(mmid)
# %%
mid = mid.drop("readmitted", axis=1)
# %%
# PCA降维
# 先标准化再降维
df_number = mid.select_dtypes(exclude=['object']).reset_index(drop=True)
# df_number_normalized = (df_number - df_number.mean()) / df_number.std()
mid_normalized = (df_number - df_number.min()) / (df_number.max() - df_number.min())
pca = PCA(n_components=22)
out = pca.fit(mid_normalized)
out = out.transform(mid_normalized)
count=0
count1=0
l=[]
for i in range(22):
    count=count+np.var(out[:,i])
for i in range(22):
    print(np.var(out[:,i])/count)
    count1=count1+np.var(out[:,i])/count
    print(count1)
    print(i)
    plt.scatter(i, np.var(out[:,i]))
    l.append([i, np.var(out[:,i])])
l=np.array(l)
plt.plot(l[:,0],l[:,1])
plt.show()
# %%
# # 在主成分上的投影
df_number = mid.select_dtypes(exclude=['object']).reset_index(drop=True)
# df_number_normalized = (df_number - df_number.mean()) / df_number.std()
mid_normalized = (df_number - df_number.min()) / (df_number.max() - df_number.min())
pca = PCA(n_components=2)
out = pca.fit(mid_normalized)
out = out.transform(mid_normalized)
out[:,1]=np.abs(out[:,1])
plt.scatter(out[:, 0], out[:, 1],)
plt.show()

print(out.shape)
# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
# k-means聚类遍历
l=[]
for i in range(2,12):
    print(i)
    km=KMeans(n_clusters=i)
    clf1 = km.fit_predict(out)
    s = silhouette_score(out, clf1)
    print("The silhouette_score= {}".format(s))
    l.append([i,s])
    plt.scatter(i, s)
    #n_s_bigger_than_zero = (silhouette_samples(out, clf1) > 0).sum()
    #print("{}/{}\n".format(n_s_bigger_than_zero, out.shape[0]))
l=np.array(l)
plt.plot(l[:,0],l[:,1])
plt.show()
# %%
# K-means聚类
#out=pd.read_csv("output4.csv")
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, silhouette_samples
# k-means聚类
km=KMeans(n_clusters=3)

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
#n_s_bigger_than_zero = (silhouette_samples(out, clf1) > 0).sum()
#print("{}/{}\n".format(n_s_bigger_than_zero, out.shape[0]))
# %%
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.metrics import silhouette_score, silhouette_samples
plt.figure(figsize=(20,6))
Z = linkage(out, method='ward', metric='euclidean')
p = dendrogram(Z, 0)
df=pd.DataFrame(Z)
df.to_csv("csvoutput/out.csv")
plt.show()
# %%
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
ac.fit(out)
clf2 = ac.fit_predict(out)

plt.scatter(out[:,0],out[:,1], c=clf2)
plt.show()
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
import scipy.cluster.hierarchy as sch
df=pd.read_csv("csvoutput/out.csv")
l=[df["0"],df["1"],df["2"],df["3"]]
l=np.array(l)
l=l.T
clf2= sch.fcluster(l,130,"distance") 
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
#n_s_bigger_than_zero = (silhouette_samples(out, clf2) > 0).sum()
#print("{}/{}\n".format(n_s_bigger_than_zero, out.shape[0]))

# %%
# 聚类结果分析可视化
mid=pd.read_csv("csvoutput/output3.csv")
# 标准化
df_number = mid.select_dtypes(exclude=['object']).reset_index(drop=True)
# df_number_normalized = (df_number - df_number.mean()) / df_number.std()
mid = (df_number - df_number.min()) / (df_number.max() - df_number.min())
mid["label"]=clf1
#mid.sort_values(by="label", inplace=True, ascending=False)
grouped=mid.groupby(mid["label"])
print(grouped.mean())
x=range(7) 
label_list=[]
mea=grouped.mean()
count=0
for i in mea:
    count=count+1
    if count>=3:
        label_list.append(i)
    if count==9:
        break
rects1 = plt.bar(x, height=mea[label_list].iloc[0], width=0.2, alpha=0.8, color='red', label="一")
rects2 = plt.bar([i + 0.2 for i in x], height=mea[label_list].iloc[1], width=0.2, color='yellow', label="二")
rects2 = plt.bar([i + 0.4 for i in x], height=mea[label_list].iloc[2], width=0.2, color='blue', label="二")
plt.xticks([index + 0.2 for index in x], label_list)
plt.xlabel("Attributes")
plt.title("Cluster feature means")
plt.show()
# %%
