import pandas as pd
from textrank4zh import TextRank4Keyword

df = pd.read_csv("../../data/同洲媒资库元数据_sample.csv", index_col='ID')
print(df['TITLE_FULL'])
keywords = []
for i in df['TITLE_FULL']:
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=i, lower=True, window=3, pagerank_config={'alpha': 0.85})
    for item in tr4w.get_keywords(len(i)/5, word_min_len=2):
        keywords.append(item.word)
        print(item.word, item.weight, type(item.word))
distinct_keywords = list(set(keywords))
dictionary = []
for dk in distinct_keywords:
    dic = {'word': dk, 'times': 0}
    dictionary.append(dic)
for k in keywords:
    for d in dictionary:
        if d['word'] == k:
            d['times'] += 1
            break
df2 = pd.DataFrame(dictionary)
df2.to_csv("../../data/keywords2.csv")
