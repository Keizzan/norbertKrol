# Import libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import joblib

# Import data from csv
data = pd.read_csv('../data/cleaned_data.csv')

# Extract findings and convert to list
findings = data['findings'].tolist()

# Create vectorizer
# Ignore terms that appear more than 80%
# Define custom stopwords for the vectorizer
custom_stopwords = ['is', 'are', 'the', 'of', 'or', 'and', 'without', 'there', 'redacted', 'no', 'in']
vectorizer = TfidfVectorizer(max_df=.8, ngram_range=(1,2), stop_words=custom_stopwords)

# Fit/transform findings
tfidf_matrix = vectorizer.fit_transform(findings)

print('TF-IDF matrix created.')
# Retrieve 100 top words
count_vectorizer = CountVectorizer(max_df=.9, min_df=2, stop_words=custom_stopwords)
count_matrix = count_vectorizer.fit_transform(findings)

print('Retrieving top 100 words.')
word_counts = count_matrix.sum(axis=0).A1
words = count_vectorizer.get_feature_names_out()
word_freq = sorted(zip(words,word_counts), key=lambda x: x[1], reverse=True)
top_words = list(word for word, _ in word_freq[:100])

# Path
path = '../model/tfidf/'

# Save trained model 
joblib.dump(vectorizer, path + 'tfidf_vectorizer.joblib')
joblib.dump(tfidf_matrix, path + 'tfidf_matrix.joblib')
joblib.dump(top_words, path + 'top_words.joblib')
print('TF-IDF model and top words saved.')