import pandas as pd
import gzip
from tqdm import tqdm

def parse(path):
  g = gzip.open(path, 'rb')
  for l in tqdm(g):
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF(r'.\data\reviews_Electronics.json.gz')

df.to_csv(r'.\data\amazon_electronics_review.csv', sep='\t', index=False)