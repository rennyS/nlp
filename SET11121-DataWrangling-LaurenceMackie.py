#Author : Laurence Mackie - 40170826
#SET11121 - Data Wrangling
#Logistical Regression Method

import pandas as pds
import re, nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import random

""" 
Files were changed from being .txt to .csv just by changing the file extension in folder, not through any code
(right click and rename from training.txt to training.csv)
"""



######### define local file names
#Load Data
test_data = 'testing.csv'
train_data = 'training.csv'

#pandas dataframe for easy access to data
test_data_dataframe = pds.read_csv(test_data, header=None, delimiter="\t", quoting=3)
test_data_dataframe.columns = ["Text"]

train_data_dataframe = pds.read_csv(train_data,header=None, delimiter="\t", quoting=3)
train_data_dataframe.columns = ["Sentiment", "Text"]

#########
# Function to stem words into their root structure
#

wordstem = PorterStemmer()
def stem_tokens(tokens, wordstem):
    stemmer = []
    for item in tokens:
        stemmer.append(wordstem.stem(item))
    return stemmer

########
#Tokenize
#

def tokenizer(text):
    # remove things that are not letters
    text = re.sub("[^a-zA-Z]", " ", text)
    #tokenize the sentences into words
    tokens = nltk.word_tokenize(text)
    #stem the words
    stems = stem_tokens(tokens, wordstem)
    return stems
##

vectorize = CountVectorizer(analyzer='word', stop_words='english',tokenizer=tokenizer,lowercase=True,max_features=87,
                            ngram_range=(1, 1))

corp_data_features = vectorize.fit_transform(train_data_dataframe.Text.tolist() + test_data_dataframe.Text.tolist())
corp_data_features_nad = corp_data_features.toarray()

# train,test split and exclude unlabeled (testing.csv) data
X_train, X_test, y_train, y_test = train_test_split(
    corp_data_features_nad[0:len(train_data_dataframe)],
    train_data_dataframe.Sentiment,
    train_size=.80,
    random_state=1434)

############################################
# Train the classifier with the train,test split

log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)

# Predict function for evaluation
y_pred = log_model.predict(X_test)

# Display classification report as the evaluation metric
print(classification_report(y_test, y_pred))
print "The accuracy score is {:.2%} ".format(accuracy_score(y_test, y_pred))

#print(confusion_matrix(y_test, y_pred))

# retrain classifier with all training data and use for sentiment classification for the testing data
log_model = LogisticRegression()
log_model = log_model.fit(X=corp_data_features_nad[0:len(train_data_dataframe)], y=train_data_dataframe.Sentiment)

# get predictions that can be used with test data
test_pred = log_model.predict(corp_data_features_nad[len(train_data_dataframe):])

# sample some of the test data sentences with appended sentiment
sample = random.sample(xrange(len(test_pred)), 20)


for text, sentiment in zip(test_data_dataframe.Text[sample], test_pred[sample]):
    print sentiment, text

