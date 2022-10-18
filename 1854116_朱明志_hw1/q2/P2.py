# %%
# 引入包
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics import mutual_info_score
# 数据导入
input = pd.read_csv("data/sim_data.csv")
npy = np.array([input["assets"], input["liabilities"],
                  input["buffer"], input["weights"]])
npy=npy.T
# %%
# 检验数据
print(npy[0])
print(npy[1])
print()
# %%
# 输出距离
print('Euclidean Distances:', pdist([npy[0], npy[1]]),'')
print('Standardized Euclidean Distances:', pdist([npy[0], npy[1]], 'seuclidean'))
print('Manhattan Distances:', pdist([npy[0], npy[1]], 'cityblock'))
print('Chebyshev Distances:', pdist([npy[0], npy[1]], 'chebyshev'))
print('Canberra Distances:', pdist([npy[0], npy[1]], 'canberra'))
print()
print('Cosine Similarity:', 1-pdist([npy[0], npy[1]], 'cosine'))
print('Dice Similarity:',1-pdist([npy[0], npy[1]], 'dice'))
print('Jaccard-Needham Similarity:',1-pdist([npy[0], npy[1]], 'jaccard'))
print('Kulsinski Similarity:',1-pdist([npy[0], npy[1]], 'kulsinski'))
print('Sokal-Michener Similarity:', 1-pdist([npy[0], npy[1]], 'sokalmichener'))

# %%
# 输出相似度
print('Pearson Correlation:', input["assets"].corr(input["liabilities"], 'pearson'))
print('Spearman Correlation:', input["assets"].corr(input["liabilities"], 'spearman'))
print('Mutual Information:', mutual_info_score(input["assets"], input["liabilities"]))
# %%
