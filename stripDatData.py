import pandas as pd
import gzip
import re
import time


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)


def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_cellphones.json.gz')

df = df.drop(['reviewerID', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime'], axis=1)

#This leaves us with just the ASIN, reviewText, overall, and summary
#Time to clean up the rest!
#tolower:

df['reviewText'].str.lower()

print df
