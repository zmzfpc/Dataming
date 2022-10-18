import csv
import zlib
import tqdm
import pickle
import pandas as pd

# entities = set()

# with open('data/ownthink_v2.csv', 'r', encoding='utf8') as kg:
#     for line in tqdm.tqdm(kg):
#         epv = line.strip().split(',')
#         entities.add(epv[0])


keywords = pd.read_csv('data/keywords2.csv', index_col=0)

entities = set(zlib.decompress(open('ejz.bin', 'rb').read()).decode().split(','))


def matchEntities(df):
    df['isEntity'] = 1 if df['word'] in entities else 0
    return df


keywords = keywords.apply(matchEntities, axis=1)

keywords.to_csv('data/keywords2.csv')
