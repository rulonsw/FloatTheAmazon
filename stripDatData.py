import gzip

import pandas as pd
from nltk.corpus import stopwords


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


sw = stopwords.words('english')

df = get_df('reviews_cellphones.json.gz')

print "Dropping unneeded data now..."
df = df.drop(['reviewerID', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime'], axis=1)

print "Completed!"
# This leaves us with just the ASIN, reviewText, overall, and summary
# Time to clean up the rest!

# to lower case & remove everything but whitespace and words:
print "scrubbing summaries and review text..."
df.summary = df.summary.str.lower().str.replace(r'[^\w\s]', ' ').str.replace(r'[-_]', ' ')
df.reviewText = df.reviewText.str.lower().str.replace(r'[^\w\s]', ' ').str.replace(r'[-_]', ' ')

print "Completed!"

counter = 0

# eliminate stopwords:
print "Eliminating stopwords..."
for word in sw:
    counter += 1
    print "Currently eliminating stopword # {0}".format(counter)

    df.reviewText = df.reviewText.str.replace(r'\b' + word + r'\b', '')
    df.summary = df.summary.str.replace(r'\b' + word + r'\b', '')



print df

# TODO: Determine the method of splitting the groups of star reviews (1 - 5) using the attributes from the vectorized data.
###Tools used: scikit SVC(Support vector classifier), NLTK's
# TODO: Maintain persistence between runs by using pickle.
# TODO: And here's the biggest: Implement a GUI that parses user input every 3-5 seconds, goes through the steps above with the input,
##and then predicts the review's star rating based on its content.
#
