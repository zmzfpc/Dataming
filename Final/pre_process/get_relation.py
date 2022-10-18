import pandas as pd


def judge(a, b):
    r = a / b
    if r > 0.5:
        return 1
    else:
        return 0


df = pd.read_csv("../../data/播放记录.csv", index_col='ID')
header = ['STB_ID', 'DURATION', 'WATCHTIME']
# header = ['PLAY_TIME', 'END_TIME']
df2 = df[header]
df2['WATCH'] = df2.apply(lambda x: (1 if (x['WATCHTIME'] / x['DURATION'] > 0.2) and (x['WATCHTIME'] > 10) else 0),
                         axis=1)
print(df2)
df2.to_csv("../../data/播放记录_processed.csv")
