# Amazon Rating Predictor
Thank you for viewing my GitHub repository! This project aims to investigate how effectively a multi-layer perceptron classifier is able to predict the star ratings of Amazon reviews based solely on review content. The code relies heavily on the following packages: 
- SciKit-Learn
- Pandas.io 's data manipulation suite
- Python's NLTK English stopwords corpus

# Running the Program
When the Python script first runs, it will check for existing .pkl files that hold CountVectorizer vocabulary data, an existing tf-idf transformer object, target values (1.0 - 5.0 star reviews, referred to in the program as either 'y' or 'target'), and trained classifiers (one Multi-Layer Perceptron Classifier and one Naive Bayes Classifier as a sort-of control group).

If any of these .pkl files are missing at runtime, you'll need to ensure the reviews_cellphones.json.gz file is in the same directory as the Python script so it may pull the data, train and fit the necessary models, and then save the requisite .pkl files for its next execution.

Once all of the data has loaded successfully, you will be able to set the number of cross-validation folds you'd like to divide your training data into. For easier comparison, I recommend running both the Multionmial NB and MLP classifier cross-validations together to observe the difference in runtime and accuracy. 

# Using Different Datasets to Train Neural Networks
If you'd like to use a different dataset to evaluate the effectiveness of MLP and Multinomial NB Classifiers, a large amount of historical Amazon review data is hosted at http://jmcauley.ucsd.edu/data/amazon/ . Simply download the compressed .json file, store it in your working directory, and follow the example input method outlined in lines 210 - 220 in the Python script. IMPORTANT: you'll want to delete all previous .pkl files to ensure your classifiers are trained on the data you've just downloaded. 

# Tweaking Your Neural Network
If you'd like to see what effect different variables have on the MLP Classifier (e.g., deeper layers of hidden perceptrons, an increased number of perceptrons per-layer, etc.), edit the stats on line 259. The first number on the MLP Classifier denotes the number of perceptrons per-layer, and the second indicates the depth of the network. 

## A Word of Warning
MLP Classifiers use back-propagation to fit data. Back-propagation is a costly process that may slow program execution considerably. 
