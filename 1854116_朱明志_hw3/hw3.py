# %%
# 引入包
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.layers.core import Dense, Activation
from keras import regularizers
from sklearn.linear_model import LogisticRegressionCV
from keras.models import Sequential
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight
import pandas as pd
from tensorflow.python.keras.engine.base_layer import AddMetric
from tensorflow.python.keras.metrics import Recall
plt.style.use('ggplot')
# %%
# 导入数据
input1 = pd.read_csv("data/基本信息.csv", encoding="gbk")
input3 = pd.read_csv("data/透析中记录.csv", encoding="gbk")
input2 = pd.read_csv("data/血透治疗单.csv", encoding="gbk")
# %%
for i in input1:
    print(input1[i].value_counts())
print("----------")
for i in input2:
    print(input2[i].value_counts())
for i in input3:
    print(input3[i].value_counts())
# %%
# 文件1编码转换
input1.replace("男", 1, inplace=True)
input1.replace("女", 0, inplace=True)
input1["birthDay"] = input1["birthDay"].apply(
    lambda x: 2019-int(x.split("-")[0]))
input1.to_csv("outputcsv/mid1.csv", index=None,
              float_format='%.3f', encoding="gbk")
# %%
# 文件2聚合
mid2 = input2.groupby(["病人id", "examineDate"]).transform('sum')
mid2["病人id"] = input2["病人id"]
mid2.drop_duplicates(subset="参数记录", inplace=True, keep="first")
input2 = mid2
input2.to_csv("outputcsv/mid2.csv", index=None,
              float_format='%.3f', encoding="gbk")
# %%
# 文件3处理
input3 = input3.drop("收缩压", axis=1)
input3.replace("", np.nan, inplace=True)
input3.dropna(inplace=True)

input3.to_csv("outputcsv/mid3.csv", index=None,
              float_format='%.3f', encoding="gbk")
# %%
# 读入数据
# %%
# merge处理
mid1 = pd.read_csv("outputcsv/mid1.csv", encoding="gbk")
mid2 = pd.read_csv("outputcsv/mid2.csv", encoding="gbk")
mid3 = pd.read_csv("outputcsv/mid3.csv", encoding="gbk")
#input3 = input3.drop(["病人id", "记录时间"], axis=1)
# %%
mid4 = pd.merge(mid3.drop("病人id", axis=1), mid2, on="参数记录")
mid4.to_csv("outputcsv/mid4.csv", index=None,
            float_format='%.3f', encoding="gbk")
# %%
out = pd.merge(mid4, mid1, on="病人id")
out = out.rename(columns={'birthDay': '年龄'})
out.to_csv("outputcsv/out.csv", index=None,
           float_format='%.3f', encoding="gbk")

# %%
out = pd.read_csv("outputcsv/out.csv", encoding="gbk")
plt.matshow(out.corr(method="pearson"))
plt.colorbar()
plt.show()
# %%
# 近零方差属性筛查
l = []
la = []
for i in out:
    la.append(i[0:1])
    l.append(out[i].value_counts().iloc[0])
x = range(len(l))
plt.bar(x, height=l, width=0.4, alpha=0.8, label="数量最大值")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.ylabel("出现次数")
plt.xticks([index for index in x], la)
plt.xlabel("属性")
plt.title("近零方差变量统计")
plt.show()
# %%
# 特征选取
fea = out.drop(["电导度", "透析液温度"], axis=1)


# %%
# 三种分类标签生成函数定义


def sbp1(df):
    n = len(df)
    series = pd.Series(0, dtype=int).reindex(df.index, method='pad')
    for i in range(n-1):
        sbpt = df['收缩压数值'].iloc[i]
        sbpt1 = df['收缩压数值'].iloc[i+1]
        if sbpt1 < 90.0:
            series.iloc[i] = 1
    series.iloc[n-1] = np.nan
    return series


def sbp2(df):
    n = len(df)
    series = pd.Series(0, dtype=int).reindex(df.index, method='pad')
    for i in range(n - 1):
        sbpt = df['收缩压数值'].iloc[i]
        sbpt1 = df['收缩压数值'].iloc[i+1]
        if sbpt-sbpt1 > 25.0:
            series.iloc[i] = 1
    series.iloc[n-1] = np.nan
    return series


def sbp3(df):
    n = len(df)
    series = pd.Series(0, dtype=int).reindex(df.index, method='pad')
    for i in range(n-1):
        sbpt = df['收缩压数值'].iloc[i]
        sbpt1 = df['收缩压数值'].iloc[i+1]
        if 0.75*sbpt > sbpt1:
            series.iloc[i] = 1
    series.iloc[n-1] = np.nan
    return series


def add123(df):
    df['sbp1'] = sbp1(df)
    df['sbp2'] = sbp2(df)
    df['sbp3'] = sbp3(df)
    return df


# %%
fea = fea.groupby(['病人id', '参数记录']).apply(add123)
fea = fea.dropna()
fea.to_csv("outputcsv/fea.csv", index=None,
           float_format='%.3f', encoding="gbk")

# %%
for i in fea:
    print(fea[i].value_counts())


# %%
# 再次读入数据
# 提取原始12列属性中的特征
fea = pd.read_csv("outputcsv/fea.csv", encoding="gbk")
fea = fea.drop("病人id", axis=1)
fea = fea.drop("参数记录", axis=1)
fea = fea.drop("记录时间", axis=1)
fea["收缩压数值"] = fea["收缩压数值"]-fea["透析前收缩压"]
fea = fea.drop("透析前收缩压", axis=1)
print(fea.shape)

# %%
# 自定义的优化recall函数


def getRecall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # TP
    P = K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P-TP  # FN=P-TP
    recall = TP / (TP + FN + K.epsilon())  # TP/(TP+FN)
    return recall

# %%
# 热编码实现


def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))
# %%
# 模型生成函数


def Simplemodelcreate():
    model = Sequential()
    model.add(Dense(28, input_shape=(7,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()
    return model
# %%
# 数据平衡权重生成


def bal(y):
    class_weight1 = class_weight.compute_class_weight('balanced',
                                                      np.unique(y),
                                                      y)
    cw = dict(enumerate(class_weight1))
    return cw


# %%
# 单层神经网络模拟实现逻辑回归算法
X = fea.values[:, 0:7]
y1 = fea.values[:, 7]
y2 = fea.values[:, 8]
y3 = fea.values[:, 9]
# Make one -hot encoder
train_y_ohe1 = one_hot_encode_object_array(y1)
train_y_ohe2 = one_hot_encode_object_array(y2)
train_y_ohe3 = one_hot_encode_object_array(y3)
# 平衡数据集
cw1 = bal(y1)
cw2 = bal(y2)
cw3 = bal(y3)
model1 = Simplemodelcreate()
model1.compile(loss='binary_crossentropy', metrics=[
    'accuracy', getRecall], optimizer=RMSprop(lr=0.0001))
model2 = Simplemodelcreate()
model2.compile(loss='binary_crossentropy', metrics=[
    'accuracy', getRecall], optimizer=RMSprop(lr=0.0001))
model3 = Simplemodelcreate()
model3.compile(loss='binary_crossentropy', metrics=[
    'accuracy', getRecall], optimizer=RMSprop(lr=0.0001))
history1 = model1.fit(X, train_y_ohe1, verbose=1, epochs=4,
                      batch_size=9, validation_split=0.2, shuffle=True, class_weight=cw1)

history2 = model2.fit(X, train_y_ohe2, verbose=1, epochs=4,
                      batch_size=9, validation_split=0.2, shuffle=True, class_weight=cw2)

history3 = model3.fit(X, train_y_ohe3, verbose=1, epochs=4,
                      batch_size=9, validation_split=0.2, shuffle=True, class_weight=cw3)

lr1 = model1.predict(X, batch_size=9, verbose=1, steps=None)
lr2 = model2.predict(X, batch_size=9, verbose=1, steps=None)
lr3 = model3.predict(X, batch_size=9, verbose=1, steps=None)
model1.save("models/LRforsbplabel-1.h5")
model2.save("models/LRforsbplabel-2.h5")
model3.save("models/LRforsbplabel-3.h5")
print("model saved!")
# %%
# 模型生成函数


def SVMmodelcreate():
    model = Sequential()
    model.add(Dense(7, input_shape=(7,), kernel_regularizer=regularizers.l2(0.5)))
    model.add(Activation('linear'))
    model.add(Dense(2, kernel_regularizer=regularizers.l2(0.5)))
    model.add(Activation('softmax'))
    model.summary()
    return model
# %%
# 自定义损失函数模拟svm分类器


def categorical_squared_hinge(y_true, y_pred):
    y_true = 2. * y_true - 1  
    # trans [0,1] to [-1,1]，注意这个，svm类别标签是-1和1
    # hinge loss，参考keras自带的hinge loss
    vvvv = K.maximum(1. - y_true * y_pred, 0.)
    vvv = K.square(vvvv)
    # 文章《Deep Learning using Linear Support Vector Machines》有进行平方
    vv = K.sum(vvv, 1, keepdims=False)  
    # axis=len(y_true.get_shape()) - 1
    v = K.mean(vv, axis=-1)
    return v


# %%
# 使用Keras实现svm分类器
X = fea.values[:, 0:7]
y1 = fea.values[:, 7]
y2 = fea.values[:, 8]
y3 = fea.values[:, 9]
# Make one -hot encoder
train_y_ohe1 = one_hot_encode_object_array(y1)
train_y_ohe2 = one_hot_encode_object_array(y2)
train_y_ohe3 = one_hot_encode_object_array(y3)
cw1 = bal(y1)
cw2 = bal(y2)
cw3 = bal(y3)
model1 = SVMmodelcreate()
model1.compile(optimizer=RMSprop(lr=0.0001), loss=[
               categorical_squared_hinge], metrics=['accuracy', getRecall])
model2 = SVMmodelcreate()
model2.compile(optimizer=RMSprop(lr=0.0001), loss=[
               categorical_squared_hinge], metrics=['accuracy', getRecall])
model3 = SVMmodelcreate()
model3.compile(optimizer=RMSprop(lr=0.0001), loss=[
               categorical_squared_hinge], metrics=['accuracy', getRecall])
history1 = model1.fit(X, train_y_ohe1, verbose=1, epochs=8,
                      batch_size=50, validation_split=0.2, shuffle=True)

# history1 = model1.fit(X, train_y_ohe1, verbose=1, epochs=8,
#                    batch_size=50, validation_split=0.2, shuffle=True,class_weight=cw1)
history2 = model2.fit(X, train_y_ohe2, verbose=1, epochs=8,
                      batch_size=50, validation_split=0.2, shuffle=True)
# history2 = model2.fit(X, train_y_ohe2, verbose=1, epochs=8,
#                    batch_size=50, validation_split=0.2, shuffle=True,class_weight=cw2)
history3 = model3.fit(X, train_y_ohe3, verbose=1, epochs=8,
                      batch_size=50, validation_split=0.2, shuffle=True)
# history3 = model3.fit(X, train_y_ohe3, verbose=1, epochs=8,
#                    batch_size=50, validation_split=0.2, shuffle=True,class_weight=cw3)
svm1 = model1.predict(X, batch_size=50, verbose=1, steps=None)
svm2 = model2.predict(X, batch_size=50, verbose=1, steps=None)
svm3 = model3.predict(X, batch_size=50, verbose=1, steps=None)
model1.save("models/SVMforsbplabel-1.h5")
model2.save("models/SVMforsbplabel-2.h5")
model3.save("models/SVMforsbplabel-3.h5")
print("model saved!")
# %%
# 多层神经网络
def ANNmodelcreate():
    model = Sequential()
    model.add(Dense(144, input_shape=(7,),activation=LeakyReLU()))
    model.add(Dense(72,activation=LeakyReLU()))
    model.add(Dense(36,activation=LeakyReLU()))
    model.add(Dense(18,activation=LeakyReLU()))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()
    return model


# %%
# 自己搭建多层神经网络
X = fea.values[:, 0:7]
y1 = fea.values[:, 7]
y2 = fea.values[:, 8]
y3 = fea.values[:, 9]
# Make one -hot encoder
ep=6
train_y_ohe1 = one_hot_encode_object_array(y1)
train_y_ohe2 = one_hot_encode_object_array(y2)
train_y_ohe3 = one_hot_encode_object_array(y3)
cw1 = bal(y1)
cw2 = bal(y2)
cw3 = bal(y3)
model1 = ANNmodelcreate()
model1.compile(loss='binary_crossentropy', metrics=[
    'accuracy', getRecall], optimizer="adam")
model2 = ANNmodelcreate()
model2.compile(loss='binary_crossentropy', metrics=[
    'accuracy', getRecall], optimizer="adam")
model3 = ANNmodelcreate()
model3.compile(loss='binary_crossentropy', metrics=[
    'accuracy', getRecall], optimizer="adam")

history1 = model1.fit(X, train_y_ohe1, verbose=1, epochs=ep,
                    batch_size=20, validation_split=0.2, shuffle=True, class_weight=cw1)

history2 = model2.fit(X, train_y_ohe2, verbose=1, epochs=ep,
                     batch_size=20, validation_split=0.2, shuffle=True, class_weight=cw2)

history3 = model3.fit(X, train_y_ohe3, verbose=1, epochs=ep,
                     batch_size=20, validation_split=0.2, shuffle=True, class_weight=cw3)
ann1 = model1.predict(X, batch_size=20, verbose=1, steps=None)
ann2 = model2.predict(X, batch_size=20, verbose=1, steps=None)
ann3 = model3.predict(X, batch_size=20, verbose=1, steps=None)
model1.save("models/ANNforsbplabel-1.h5")
model2.save("models/ANNforsbplabel-2.h5")
model3.save("models/ANNforsbplabel-3.h5")
print("model saved!")
# %%
# 结果可视化


def show(history):
    print(history.history)
    l = []
    ll = []
    for i in history.history:
        ll.append(i)
        l.append(history.history[i])
    l = np.array(l).T
    # print(l)
    j = range(1, 1+len(l))
    count = 0
    for i in ll:
        plt.plot(j, l[:, count], label=i)
        count = count+1
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.ylabel("训练参数值")
    plt.xlabel("训练轮数")
    plt.title("训练结果可视化")
    plt.legend()
    plt.show()


# %%
# 训练可视化
show(history1)
show(history2)
show(history3)
# %%
# 矩阵标签转换为标签列


def trans(res):
    l = []
    for i in res:
        if i[0] > i[1]:
            l.append(0)
        else:
            l.append(1)
    l = np.array(l)
    return l


# %%
# 结果文件生成
fea = pd.read_csv("outputcsv/fea.csv", encoding="gbk")
res = pd.DataFrame()
res["病人id"] = fea["病人id"]
res["记录时间"] = fea["记录时间"]
res["真实分类标签1"] = fea["sbp1"]
res["真实分类标签2"] = fea["sbp2"]
res["真实分类标签3"] = fea["sbp3"]
res["LR分类器预测标签1"] = trans(lr1)
res["LR分类器预测标签2"] = trans(lr2)
res["LR分类器预测标签3"] = trans(lr3)
res["SVM分类器预测标签1"] = trans(svm1)
res["SVM分类器预测标签2"] = trans(svm2)
res["SVM分类器预测标签3"] = trans(svm3)
res["ANN分类器预测标签1"] = trans(ann1)
res["ANN分类器预测标签2"] = trans(ann2)
res["ANN分类器预测标签3"] = trans(ann3)

res.to_csv("outputcsv/result.csv", index=None,
           float_format='%.3f', encoding="gbk")
# %%

# %%
for i in res:
    print(res[i].value_counts())
# %%
res = pd.DataFrame()
res["真实分类标签1"] = fea["sbp1"]
res["真实分类标签2"] = fea["sbp2"]
res["真实分类标签3"] = fea["sbp3"]
label_list = ["真实分类标签1", "真实分类标签2", "真实分类标签3"]
x = range(3)
plt.style.use('ggplot')
for i in res:
    print(res[i].value_counts())
    rects1 = plt.bar(x, height=res[i].value_counts(
    ).iloc[0], width=0.2, alpha=0.8,  label="低风险")
    rects2 = plt.bar([i + 0.2 for i in x],
                     height=res[i].value_counts().iloc[1], width=0.2, label="高风险")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xticks([index + 0.1 for index in x], label_list)
plt.xlabel("标签")
plt.ylabel("数量")
plt.title("每种标签的数量直方图")
plt.show()

# %%
