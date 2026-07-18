# for data manipulation
import pandas as pd
# for preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# model, tuning and metrics
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
# model serialization
import joblib
import os
# Hugging Face authentication and uploads
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-wellness-training")

api = HfApi(token=os.getenv("HF_TOKEN"))

# Load the train/test splits from the Hugging Face dataset space
base = "hf://datasets/YOUR_HF_USERNAME/tourism-wellness-package"
Xtrain = pd.read_csv(f"{base}/Xtrain.csv")
Xtest = pd.read_csv(f"{base}/Xtest.csv")
ytrain = pd.read_csv(f"{base}/ytrain.csv").values.ravel()
ytest = pd.read_csv(f"{base}/ytest.csv").values.ravel()
print("Loaded train/test data from the Hub.")

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Base classifier with class-imbalance handling
scale_pos_weight = float((ytrain == 0).sum() / (ytrain == 1).sum())
xgb_model = xgb.XGBClassifier(
    random_state=42, n_jobs=-1, eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
)

# Hyperparameter grid: 2 x 2 x 2 x 2 x 2 = 32 combinations, x 3-fold CV = 96 fits
# (kept compact so the GitHub Actions model-training job finishes reliably)
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline, param_grid, cv=3, n_jobs=-1, scoring="f1"
    )
    grid_search.fit(Xtrain, ytrain)

    # Log each parameter set as a nested run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])

    # Log best params
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Evaluate on BOTH train and test sets (train metrics enable an over-fit check)
    y_pred_train = best_model.predict(Xtrain)
    y_proba_train = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_test = best_model.predict(Xtest)
    y_proba_test = best_model.predict_proba(Xtest)[:, 1]
    metrics = {
        "train_accuracy": accuracy_score(ytrain, y_pred_train),
        "train_precision": precision_score(ytrain, y_pred_train),
        "train_recall": recall_score(ytrain, y_pred_train),
        "train_f1": f1_score(ytrain, y_pred_train),
        "train_roc_auc": roc_auc_score(ytrain, y_proba_train),
        "test_accuracy": accuracy_score(ytest, y_pred_test),
        "test_precision": precision_score(ytest, y_pred_test),
        "test_recall": recall_score(ytest, y_pred_test),
        "test_f1": f1_score(ytest, y_pred_test),
        "test_roc_auc": roc_auc_score(ytest, y_proba_test),
    }
    mlflow.log_metrics(metrics)
    print("Train metrics:", {k: round(v, 4) for k, v in metrics.items() if k.startswith("train")})
    print("Test metrics: ", {k: round(v, 4) for k, v in metrics.items() if k.startswith("test")})
    # Over-fit check: a large positive gap between train and test F1 indicates over-fitting
    print(f"Over-fit check (train_f1 - test_f1): {metrics['train_f1'] - metrics['test_f1']:.4f}")

    # Save the best model locally
    model_path = "best_tourism_wellness_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as: {model_path}")

    # Register the model in the Hugging Face model hub
    repo_id = "YOUR_HF_USERNAME/tourism-wellness-model"
    repo_type = "model"
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{repo_id}' not found. Creating new model repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Uploaded model to the Hugging Face model hub.")
