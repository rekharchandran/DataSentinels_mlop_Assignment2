# Email Spam Detection
### Classify emails as spam or not using NLP techniques
### About the dataset
The SMS Spam Collection v.1 is a set of SMS messages that have been collected and labeled as either spam or not spam. This dataset contains 5574 English, real, and non-encoded messages. The SMS messages are thought-provoking and eye-catching. The dataset is useful for mobile phone spam research

This dataset is used to train a machine learning model to classify SMS messages as spam or not spam.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Display basic information about the dataset
print(df.info())
print(df.head())
# Keep only the necessary columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Check class distribution
print(df['label'].value_counts())
# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text cleaning and preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
corpus = []

for msg in df['message']:
    # Remove special characters and convert to lower case
    msg = re.sub('[^a-zA-Z]', ' ', msg).lower()
    msg = msg.split()
    
    # Lemmatization and stopword removal
    msg = [lemmatizer.lemmatize(word) for word in msg if word not in stopwords.words('english')]
    msg = ' '.join(msg)
    corpus.append(msg)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)

# Performance metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
def predict_spam(message):
    msg = re.sub('[^a-zA-Z]', ' ', message).lower()
    msg = msg.split()
    msg = [lemmatizer.lemmatize(word) for word in msg if word not in stopwords.words('english')]
    msg = ' '.join(msg)
    vect = vectorizer.transform([msg]).toarray()
    vect = tfidf.transform(vect).toarray()
    prediction = model.predict(vect)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Test with a new message
print(predict_spam("Congratulations! You've won a rs 100000 reliance gift card. Call now!"))