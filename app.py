from flask import Flask, render_template, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.artifacts
import joblib
import os

app = Flask(__name__)

# MLflow experiment configuration (ensure this matches your MLflow setup)
EXPERIMENT_NAME = "Email Spam Detection"
client = MlflowClient()

def load_latest_artifact_model():
    """
    Loads the best model and TF-IDF vectorizer from the latest MLflow run by
    downloading the logged artifacts.
    Assumes the artifacts are saved as "model.pkl" and "vectorizer.pkl".
    """
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

    # Download artifacts from the best run to a local temporary directory
    artifact_dir = mlflow.artifacts.download_artifacts(run_id=run_id)
    
    # Build full paths to the artifact files
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
    Uses the loaded TF-IDF vectorizer to transform the text and the saved model to predict.
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
    Triggers retraining of the model and returns the new best parameters.
    The retrain_model() function should handle re-training, logging new artifacts to MLflow,
    and then return the new model and vectorizer.
    """
    try:
        global best_model, best_vectorizer, best_params
        new_model, new_vectorizer, new_best_params = retrain_model()  # You need to implement retrain_model()
        best_model = new_model
        best_vectorizer = new_vectorizer
        best_params = new_best_params
        return jsonify({"message": "Training completed", "best_params": best_params})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Renders the interactive web page.
    """
    prediction = None
    # The interactive page now uses AJAX for API calls,
    # so we don't need to handle POST here.
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
