# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# %%
df1 = pd.read_csv("../../data/同洲媒资库元数据_sample.csv", index_col='ID')

# 完全没用的列
delete = [
    'DETAIL_URI', 'DISPLAY_FLAGS', 'END_DATE_TIME', 'FAVOR_RATING', 'FLAG_IMAGE_URL', 'IMAGE_LOCATION', 'INITIAL_LETTER', 'IS_PACKAGE', 'MD5', 'PREVIEW_PROVIDER_ID', 'PRIVIDER_NAME', 'RECOMMAND_RATING', 'RECOMMAND_TIMES',
    'RECOMMENDATION_LEVEL', 'SEARCH_ABLE', 'SECOND_TITLE_FULL', 'SERVICE_ID', 'SERVICE_TYPE', 'SITE_FLAG', 'START_DATE_TIME', 'STATUS', 'SUMMAR_MEDIUM', 'SUMMARV_SHORT', 'TITLE_BRIEF', 'TITLE_FULL', 'VIDEO_TYPE', 'VIEW_LEVEL'
]
df1.drop(delete, axis=1, inplace=True)
df1.to_csv("../../data/同洲媒资库元数据_processed.csv")


df2 = pd.read_csv("../../data/播放记录.csv", index_col='ID')

# 完全没用的列
delete = [
    'Unnamed: 0', 'index', 'ASSET_TYPE', 'CP', 'ELAPSED', 'NAME', 'OPK', 'PORTAL_VER', 'PRICE', 'PROVIDER_ID', 'q', 's', 'VIRTUAL_OPK'
]
df2.drop(delete, axis=1, inplace=True)
df2.to_csv("../../data/播放记录_processed.csv")

# %%
