from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "NineKnox/tourism-wellness-prediction"
repo_type = "space"

# Create the Space if it does not exist.
# NOTE: Hugging Face now requires a PRO subscription to host Docker/Gradio Spaces
# on the free cpu-basic tier, so we use the free **Streamlit SDK**. It runs
# app.py + requirements.txt directly (no Dockerfile needed at runtime).
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new Space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, space_sdk="gradio", private=False)
    print(f"Space '{repo_id}' created.")

# Upload the app + dependencies. Skip the Dockerfile: the Streamlit SDK does not
# use it, and uploading it could make the Space try to build as Docker (PRO-only).
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",
    ignore_patterns=["Dockerfile"],
)
print("Uploaded app.py + requirements.txt to the Hugging Face Streamlit Space.")
