from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "NineKnox/tourism-wellness-prediction"
repo_type = "space"

# Create the Space if it does not exist.
# HF removed the Streamlit SDK and made Docker / CPU-basic PRO-only, so the free
# frontend uses the **Gradio** SDK (free Gradio Spaces run on ZeroGPU hardware,
# and app.py is decorated with @spaces.GPU to satisfy the ZeroGPU scheduler).
# NOTE: creating the Space from the HF *website* is the reliable path -- it lands
# on the free ZeroGPU tier automatically. This create call is only a CI fallback.
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new Space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, space_sdk="gradio", private=False)
    print(f"Space '{repo_id}' created.")

# Upload the app + dependencies. Skip the Dockerfile: the Gradio SDK does not
# use it, and uploading it could make the Space try to build as Docker (PRO-only).
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",
    ignore_patterns=["Dockerfile"],
)
print("Uploaded app.py + requirements.txt to the Hugging Face Gradio Space.")
