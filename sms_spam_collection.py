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

# Read the large CSV file
df = pd.read_csv("train.csv")

# If 'Unnamed: 0' column exists (index column), drop it
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Assuming 'sms' is the message column and 'label' is the target column
df = df[['sms', 'label']]  # Adjust column names as per your dataset

# Rename columns for clarity
df.columns = ['message', 'label']

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

# Convert 'label' to categorical (if needed)
df['label'] = df['label'].astype(int)  # Ensure it's numeric

# Check for missing values
print(df.isnull().sum())

# Feature and target columns
X = df['cleaned_message']
y = df['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train = vectorizer.fit_transform(X_train)

# Transform the test data (use the same vectorizer)
X_test = vectorizer.transform(X_test)

# Initialize the model
model = MultinomialNB()

# Enable MLflow autologging
mlflow.sklearn.autolog()

# Start an MLflow run
with mlflow.start_run():
    print("Training the model...")
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    # Classification report
    report = classification_report(y_test, y_pred, zero_division=0)
    print("Classification Report:\n", report)

    # Log the accuracy manually (optional, since autolog() already logs it)
    mlflow.log_metric("manual_accuracy", accuracy)

    # Optionally, log the classification report as an artifact
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # Log the trained model
    mlflow.sklearn.log_model(model, "model")
