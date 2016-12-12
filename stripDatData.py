import gzip
import time

import pandas as pd
from nltk.corpus import stopwords
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize


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

# words to nix from our database
sw = stopwords.words('english')


# Categories (i.e., ratings)
def prep_data(frame):
    print "Dropping data..."
    frame.drop(['reviewerID', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime', 'summary'], axis=1)
    print "Complete!\n"

    print "Scrubbing review text..."
    frame.reviewText = df.reviewText.str.lower().str.replace(r'[^\w\s]', ' ').str.replace(r'[-_]', ' ')
    print "Completed scrubbing!\n\n"

def prep_usr_input(strang):
    print "Scrubbing user review text..."
    strang = strang.lower()
    strang = strang.replace(r'[^\w\s]', ' ')
    strang = strang.replace(r'[-_]', ' ')
    print "Completed scrubbing user review!\n\n"
    return strang


def compile_data_target(frame, X, y):
    if not X:
        for i in xrange(0, frame.reviewText.size):
            X.append(df.reviewText[i])
            if i % 10000 == 0:
                print "review {0} of {1} appended.".format(i, frame.reviewText.size)
    else:
        print "String samples already present."
    if not y:
        for i in xrange(0, frame.overall.size):
            y.append(df.overall[i])
            if i % 10000 == 0:
                print "rating {0} of {1} appended.".format(i, frame.overall.size)
    else:
        print "Existing target data."


    print "Insertion/appending complete.\n\n"


def try_unpickle():
    load_state = 0
    txt_MLPC = 0
    txt_MNB = 0
    vector_vocab = 0
    tfidf = 0
    y = []
    try:
        print "Attempting to pull trained models and fit_transform data from .pkl files...\n"

        print "tf-idf..."
        tfidf = joblib.load('tfidf_data.pkl')
        print "Success!"

        print "target..."
        y = joblib.load('y_data.pkl')
        print "Success!"

        print "vectorizer vocab..."
        vector_vocab = joblib.load('vecVocab_data.pkl')
        print "Success!"

        print "Multinomial Naive Bayes..."
        txt_MNB = joblib.load('trained_MNB.pkl')
        print "Success!\n"

        load_state += 1

        print "Multilayer Perceptron Classifier..."
        txt_MLPC = joblib.load('trained_MLPC.pkl')
        print "Success!"
        load_state += 1

        if not isinstance(tfidf, (int, long)):
            load_state +=1

        if not isinstance(vector_vocab, (int, long)):
            load_state +=1

        if not isinstance(y, (int, long)):
            load_state +=1

    except IOError:
        print "Error - one or more .pkl files missing. training and creating new models.\n\n\n"

    return load_state, txt_MNB, txt_MLPC, vector_vocab, tfidf, y


def interactive(choice, txt_MNB, txt_MLPC):
    # if user would like to see a prediction based on their review data:
    if choice == 0:
        review = raw_input("Please enter your review data here.")
        review = prep_usr_input(review)
        print "Please wait..."

        #TODO: FINISH USER-INTERACTIVE EXAMPLE
        print "According to the Multinomial Naive Bayes predictor, my best guess for your star rating is: {0} ".format(
            txt_MNB.predict
        )
    # else, if user would like to perform K-fold cross-validation to observe accuracy of learning methods:
    elif choice == 1:
        print "######### K FOLD VALIDATION #########\n\n\n"
        K_folds = input("How many folds would you like to validate your data against?")

        to_test = raw_input("\n\nWhich classifier would you like to test?\n" +
                            "Please choose 1 for Naive Bayes, 2 for the MLP Classifier, or any other number for both.")

        if to_test == 1:
            start = time.clock()
            print "Naive Bayes {0}-fold cross-validation: ".format(K_folds)
            NB_scores = cross_val_score(txt_MNB, tfidf, y, cv=K_folds)
            print NB_scores
            total_time = time.clock() - start
            print "Total processing time: {0} seconds".format(total_time)

        elif to_test == 2:
            print "MLPC {0}-fold cross-validation: ".format(K_folds)
            start = time.clock()
            MLPC_scores = cross_val_score(txt_MLPC, tfidf, y, cv=K_folds)
            print MLPC_scores
            total_time = time.clock() - start
            print "Total processing time: {0} seconds".format(total_time)

        else:
            start = time.clock()
            print "Naive Bayes {0}-fold cross-validation: ".format(K_folds)
            NB_scores = cross_val_score(txt_MNB, tfidf, y, cv=K_folds)
            print NB_scores
            total_time = time.clock() - start
            print "Total processing time: {0} seconds\n\n".format(total_time)

            print "MLPC {0}-fold cross-validation: ".format(K_folds)
            normalized_tfidf = normalize(tfidf, norm='l2')
            MLPC_scores = cross_val_score(txt_MLPC, normalized_tfidf , y, cv=K_folds)
            print MLPC_scores
            total_time = time.clock() - total_time
            print "Total processing time: {0} seconds\n\n".format(total_time)
# data lists
X = []                      # actual review content
y = []                      # star ratings from JSON

load_success, txt_MNB, txt_MLPC, vec_vocab, tfidf, y = try_unpickle()

# Display what we were able to pull from external databases:
print "load status: {0}".format(load_success)
print "txt_MNB: "
print txt_MNB
print ""
print "txt_MLPC: "
print txt_MLPC
print ""
print "vec_vocab: "
print vec_vocab
print ""
print "tfidf: "
print tfidf
print ""
print "Checking target data type: "
if isinstance(y, (int, long)):
    print "Int"
else:
    print "List/Sparse/Other"


if vec_vocab != 0:
    vectorizer = CountVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 stop_words=sw,
                                 vocabulary=vec_vocab,
                                 preprocessor = None,
                                 max_features = 5000)
else:
    vectorizer = CountVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 stop_words=sw,
                                 preprocessor = None,
                                 max_features = 5000)

if load_success < 2 or vec_vocab == 0 or isinstance(tfidf, (int, long)) or isinstance(y, (int, long)):
    # cellphone review data
    print "Cellphone data..."
    df = get_df('reviews_cellphones.json.gz')
    prep_data(df)
    compile_data_target(df, X, y)

    # # electronics review data
    # print "Electronics data..."
    # df = get_df('reviews_Electronics_5.json.gz')
    # prep_data(df)
    # compile_data_target(df, X, y)



    # look at the data real quick
    print vectorizer

    print "Fitting vectorizer..."
    train_from_data = vectorizer.fit_transform(X, y)

    print train_from_data


    # Time for a tf transformer (turning occurrences to frequencies)

    # Python-ism: it's easier to ask forgiveness than to ask permission. Checking if tfidf is scalar (i.e., 0) or vector
    try:
        tfidf + 42
        print "Creating new tf-idf transformer"
        tfidf = TfidfTransformer(use_idf=True).fit_transform(train_from_data)
    except NotImplementedError:
        print "tf-idf data present. Moving on..."

    print tfidf

    print "Shape: "
    print tfidf.shape
    if load_success == 0:
        # ################ MULTINOMIAL NAIVE BAYES ######################
        txt_MNB = MultinomialNB()

        txt_MNB.fit(tfidf, y)
        print txt_MNB
        # ###############################################################

    # #################### NEURAL NETWORK ###########################
    print "Normalizing MLP Classifier data..."
    normalized_tfidf = normalize(tfidf, norm='l2')
    print "Complete!"
    txt_MLPC = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 3))
    txt_MLPC.fit(normalized_tfidf, y)
    # ###############################################################


# Save data before closing... #
print "Saving MNB..."
joblib.dump(txt_MNB, 'trained_MNB.pkl')
print "Complete!\n"

print "Saving MLPC..."
joblib.dump(txt_MLPC, 'trained_MLPC.pkl')
print "Complete!\n\n"

print "Saving vec_vocab data..."
joblib.dump(vectorizer.vocabulary, 'vecVocab_data.pkl')
print "Complete!\n\n"

print "Saving tf-idf transformation data..."
joblib.dump(tfidf, 'tfidf_data.pkl')
print "Complete!\n\n"

print "Saving target data..."
joblib.dump(y, 'y_data.pkl')
print "Complete!\n\n"

interactive(1, txt_MNB, txt_MLPC)
raw_input()

# ## SVCs don't handle vast tracts of data very well. Implementing a MLPClassifier instead.

# ## As it turns out, MLP Classifiers don't handle this much much data very well, either. I'll stick with it and
# ## keep the layers fairly shallow.

