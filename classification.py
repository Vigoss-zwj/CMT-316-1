from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas

#==========================================================================================================================================#
#                                      read bbc_news.csv
bbc_news_data = pandas.read_csv('./Part2_corpus/bbc_news.csv')
bbc_news_data.head()

#==========================================================================================================================================#
#                                    define text and label
trainDF = pandas.DataFrame()
trainDF['text'] = bbc_news_data['content']
trainDF['label'] = bbc_news_data['label']

#==========================================================================================================================================#
#                            seperate corpus into training set and testing set
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], random_state=4500)

#==========================================================================================================================================#
#                                       label encoder
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

#==========================================================================================================================================#
#                                      counter vector
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

#==========================================================================================================================================#
#                                       cross valid
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)

#==========================================================================================================================================#
#                                   word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

#==========================================================================================================================================#
#                                   n-gram tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)

#==========================================================================================================================================#
#                                 character tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x)

#==========================================================================================================================================#
#                                    define train model
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net = False):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(valid_y, predictions), metrics.classification_report(valid_y, predictions, target_names = ['business', 'entertainment', 'politics', 'sport', 'tech'])

#==========================================================================================================================================#
#                                   bayes classifier
# ----------------------------------on count vectors------------------------------------------------------------------
accuracy,classification_report = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("Bayes + Count Vector: ", accuracy)
print(classification_report)

# ----------------------------------on word level tf-idf------------------------------------------------------------------
accuracy,classification_report = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("Bayes + Word Level TF-IDF: ", accuracy)
print(classification_report)

# ---------------------------------on n-gram level tf-idf------------------------------------------------------------------
accuracy,classification_report = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("Bayes + N-gram Level TF-IDF: ", accuracy)
print(classification_report)

# -------------------------------on character level tf-idf----------------------------------------------------------------------
accuracy,classification_report = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("Bayes + Charater Level TF-IDF: ", accuracy)
print(classification_report)

#==========================================================================================================================================#
#                                linear regression classifier
# ----------------------------------on count vectors------------------------------------------------------------------
accuracy,classification_report = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR + Count Vectors: ", accuracy)
print(classification_report)

# ----------------------------------on word level tf-idf------------------------------------------------------------------
accuracy,classification_report = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR + Word Level TF-IDF: ", accuracy)
print(classification_report)

# ---------------------------------on n-gram level tf-idf------------------------------------------------------------------
accuracy,classification_report = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR + N-Gram Level Vectors: ", accuracy)
print(classification_report)

# -------------------------------on character level tf-idf----------------------------------------------------------------------
accuracy,classification_report = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR + Charater Level Vectors: ", accuracy)
print(classification_report)