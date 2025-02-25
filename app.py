from flask import Flask, render_template, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.artifacts
import joblib
import os
import pandas as pd
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# MLflow experiment configuration
EXPERIMENT_NAME = "Email Spam Detection"
client = MlflowClient()

def load_latest_artifact_model():
    """
    Loads the best model and TF-IDF vectorizer from the latest MLflow run by
    downloading the logged artifacts. Assumes the artifacts are saved as "model.pkl" and "vectorizer.pkl".
    """
    # Retrieve the default experiment (ID "0")
    # experiment = mlflow.get_experiment("0")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print("Experiment not found!")
        return None, None, {}
    experiment_id = experiment.experiment_id
    runs = client.search_runs(experiment_ids=[experiment_id],
                              order_by=["metrics.accuracy DESC"],
                              max_results=1)
    if not runs:
        print("No runs found for the experiment!")
        return None, None, {}
    best_run = runs[0]
    run_id = best_run.info.run_id

    artifact_dir = mlflow.artifacts.download_artifacts(run_id=run_id)
    model_path = os.path.join(artifact_dir, "model.pkl")
    vectorizer_path = os.path.join(artifact_dir, "vectorizer.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Artifact files not found in the downloaded directory.")
        return None, None, best_run.data.params

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except Exception as e:
        print("Error loading artifacts:", e)
        return None, None, best_run.data.params
    
    return model, vectorizer, best_run.data.params

def retrain_model(df=None):
    """
    Re-trains the model using hyperparameter tuning, logs new artifacts (model and vectorizer)
    to MLflow, and returns the new model pipeline, vectorizer, and best parameters.
    
    If a DataFrame 'df' is provided, it is used as the training data.
    Otherwise, the default dataset in "train.csv" is used.
    
    The expected dataset format is two columns: "sms" and "label".
    """
    # Use provided DataFrame or load default data
    if df is None:
        df = pd.read_csv("train.csv")
        df = df[['sms', 'label']]
        df.columns = ['message', 'label']
    else:
        # Assume incoming new data is provided as a list of records with columns: sms and label.
        # Ensure the DataFrame has the correct column names.
        df.columns = ['sms', 'label']
        df.rename(columns={'sms': 'message'}, inplace=True)
    
    # Text cleaning
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return " ".join(words)
    
    df['cleaned_message'] = df['message'].apply(clean_text)
    
    X = df['cleaned_message']
    y = df['label']
    
    # Vectorize text
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    
    # Split data into training and testing sets (e.g., 70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.30, random_state=42, shuffle=True)
    
    # Initialize model and hyperparameter grid
    model = MultinomialNB()
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
        'fit_prior': [True, False]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    
    # Start a new MLflow run for retraining
    with mlflow.start_run() as run:
        grid_search.fit(X_train, y_train)
        best_params_local = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        
        # Log parameters and metrics to MLflow
        mlflow.log_params(best_params_local)
        mlflow.log_metric("accuracy", accuracy)
        
        # Save the best model and the vectorizer using joblib.dump()
        model_filename = "model.pkl"
        vectorizer_filename = "vectorizer.pkl"
        joblib.dump(best_model, model_filename)
        joblib.dump(vectorizer, vectorizer_filename)
        
        # Log the artifacts to MLflow
        mlflow.log_artifact(model_filename)
        mlflow.log_artifact(vectorizer_filename)
        
        new_run_id = run.info.run_id
    
    # Download and load the new artifacts
    artifact_dir = mlflow.artifacts.download_artifacts(run_id=new_run_id)
    new_model_path = os.path.join(artifact_dir, model_filename)
    new_vectorizer_path = os.path.join(artifact_dir, vectorizer_filename)
    
    new_model_pipeline = joblib.load(new_model_path)
    new_vectorizer = joblib.load(new_vectorizer_path)
    
    return new_model_pipeline, new_vectorizer, best_params_local

# Load the model and vectorizer at startup
best_model, best_vectorizer, best_params = load_latest_artifact_model()

@app.route('/best_model_parameter', methods=['GET'])
def get_best_model_parameter():
    """
    Returns the best model parameters as a JSON response.
    """
    if best_params:
        return jsonify(best_params)
    else:
        return jsonify({"error": "Best model parameters not found"}), 404

@app.route('/prediction', methods=['POST'])
def api_prediction():
    """
    Expects a JSON payload with a "message" key.
    Transforms the text using the saved TF-IDF vectorizer and predicts using the saved model.
    """
    data = request.get_json(force=True)
    if 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    message = data['message']
    try:
        transformed_message = best_vectorizer.transform([message])
        prediction = best_model.predict(transformed_message)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/training', methods=['POST'])
def api_training():
    """
    Triggers retraining of the model. If new data is provided in the request payload under the key "data",
    that data will be used for training. Otherwise, the default dataset from "train.csv" is used.
    After training, the endpoint returns the new best parameters and evaluation metrics.
    """
    try:
        payload = request.json
        if payload and "data" in payload:
            # New data should be provided as a list of records, where each record has two values:
            # the sms text and the label (e.g., [["Free offer!!!", 1], ["Hi, how are you?", 0], ...])
            new_data = pd.DataFrame(payload["data"], columns=["sms", "label"])
            df = new_data
        else:
            # Use the default SMS spam dataset from train.csv
            df = pd.read_csv("train.csv")
            df = df[['sms', 'label']]
            df.columns = ['message', 'label']
        
        new_model, new_vectorizer, new_best_params = retrain_model(df)
        global best_model, best_vectorizer, best_params
        best_model = new_model
        best_vectorizer = new_vectorizer
        best_params = new_best_params
        
        return jsonify({"message": "Training completed", "best_params": best_params})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Renders an interactive web page with three columns for:
    1. Getting best model parameters,
    2. Detecting spam,
    3. Retraining the model.
    """
    prediction = None
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
