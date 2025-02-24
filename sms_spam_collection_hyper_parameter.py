import joblib
import pandas as pd
import re
import mlflow
import mlflow.sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV

mlflow.set_experiment("Email Spam Detection")

# Load your data (ensure this path is correct)
df = pd.read_csv("train.csv")

# Data Preprocessing: Clean text, split columns, etc.
df = df[['sms', 'label']]  # Select relevant columns
df.columns = ['message', 'label']  # Rename columns

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Define text cleaning function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Tokenize words
    words = text.split()
    
    # Remove stopwords and apply lemmatization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Join words back into a string
    return " ".join(words)

# Apply text cleaning to the 'message' column
df['cleaned_message'] = df['message'].apply(clean_text)

# Check the cleaned data
print(df[['message', 'cleaned_message']].head())

# Clean the message text (you should have the clean_text function from earlier)


# Split the data into features and labels
X = df['cleaned_message']
y = df['label']

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes model
model = MultinomialNB()

# Define hyperparameters for GridSearchCV
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
    'fit_prior': [True, False]
}

# Set up GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# Enable MLflow autologging
mlflow.sklearn.autolog()

# Start the MLflow run
with mlflow.start_run():
    print("Training with hyperparameter tuning...")
    
    # Perform hyperparameter tuning and train the model
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and model from GridSearchCV
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Log the best hyperparameters
    mlflow.log_params(best_params)
    
    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    # Classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)
    
    # Log the accuracy as a metric (auto-logged by MLflow)
    mlflow.log_metric("accuracy", accuracy)

    # Log the classification report as an artifact
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # Save the trained model and vectorizer to disk for inference
    model_filename = "spam_classifier_model.pkl"
    vectorizer_filename = "tfidf_vectorizer.pkl"
    
    joblib.dump(best_model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    
    # Log the saved model and vectorizer as artifacts
    mlflow.log_artifact(model_filename)
    mlflow.log_artifact(vectorizer_filename)
    
    print("Model and vectorizer saved for inference.")