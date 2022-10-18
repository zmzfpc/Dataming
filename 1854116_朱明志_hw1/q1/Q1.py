# %%
# 一个展示分类器的示例代码，并无语句调用这个分类器
import numpy as np
import cv2
img=cv2.imread("photos/img_836.jpg")
img_arr = np.asarray(img, dtype=np.double)
r_img = img_arr[:, :, 0]
g_img = img_arr[:, :, 1]
b_img = img_arr[:, :, 2]
def div(green,red,blue):
    if ((0.5*red)&(0.3*green)&(0.2*blue> 0.6)): 
        return "drink"
    else:
        return "driving"

# %%
# 导入果酒数据
# 导入鸢尾花数据
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
data=datasets.load_wine()
X=data['data']
y=data['target']
print(X)
print(y)
# %% 
#选取三个特征查看IRIS数据分布

fig=plt.figure()
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.figsize'] = (4, 4)
ax = Axes3D(fig)
for c,i,target_name in zip('>o*',[0,1,2],data.target_names):
    ax.scatter(X[y==i ,0], X[y==i, 1], X[y==i,2], marker=c, label=target_name)
ax.set_xlabel(data.feature_names[0])
ax.set_ylabel(data.feature_names[1])
ax.set_zlabel(data.feature_names[2])
ax.set_title("Wine")
#fig.savefig('Wine-show.jpg')
plt.legend()
plt.show()

# %%

# 利用PCA降维，降到二维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_p =pca.fit(X).transform(X)
ax = plt.figure()
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.figsize'] = (4, 4)
for c, i, target_name in zip("rgb", [0, 1, 2], data.target_names):
    plt.scatter(X_p[y == i, 0], X_p[y == i, 1], c=c, label=target_name)
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.title("wine-PCA")
plt.legend()
plt.show()

# %%
# 标准化后降维
from sklearn.preprocessing import StandardScaler
X=StandardScaler().fit(X).transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_p =pca.fit(X).transform(X)
ax = plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], data.target_names):
    plt.scatter(X_p[y == i, 0], X_p[y == i, 1], c=c, label=target_name)
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.title("wine-standard-PCA")
plt.legend()
plt.show()
# %%
# 有监督的LDA降维
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
# 有监督在程序中表现在X_r =lda.fit(X,y).transform(X)，用到了标签y
X_r =lda.fit(X,y).transform(X)
ax = plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], data.target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.title("LDA")
plt.legend()
plt.show()

# %%
#方差选择法，返回值为特征选择后的数据
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
data=datasets.load_iris()

#参数threshold为方差的阈值
df=pd.DataFrame(VarianceThreshold(threshold=0.6).fit_transform(data.data))

df.to_csv("csv/out1.csv", index=None, float_format='%.3f')
print("csv save!")
data1=VarianceThreshold(threshold=0.6).fit_transform(data.data)
cc=["r","g","b",]
count=0
print(data1[0])
for row in data1:
    print(row)
    plt.scatter(row[0], row[1], c=cc[data.target[count]], label=data.target[count])
    count=count+1
plt.title("Filter")
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()
# %%
#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
 
df=pd.DataFrame(RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(data.data, data.target))
df.to_csv("csv/out2.csv", index=None, float_format='%.3f')
print("csv save!")
df['counter'] = range(len(df))
for c,i in zip("bg", [0, 1] ):
    plt.scatter(df["counter"], df[i], c=c, label="Feature"+str(i))
plt.title("Wrapper")
plt.xlabel('Count')
plt.ylabel('Feature Value')
plt.legend()
plt.show()
# %%
#带L2惩罚项的逻辑回归作为基模型的特征选择
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
 
df=pd.DataFrame(SelectFromModel(LogisticRegression(penalty="l2", C=0.1)).fit_transform(data.data, data.target))

df.to_csv("csv/out3.csv", index=None, float_format='%.3f')
print("csv save!")
df['counter'] = range(len(df))
for c,i in zip("bg", [0, 1] ):
    plt.scatter(df["counter"], df[i], c=c, label="Feature"+str(i))
plt.title("Embedded")
plt.xlabel('Count')
plt.ylabel('Feature Value')
plt.legend()
plt.show()
# %%
