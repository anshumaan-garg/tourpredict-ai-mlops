from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "YOUR_HF_USERNAME/tourism-wellness-prediction"
repo_type = "space"

# Create the Space if it does not exist (Docker SDK)
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new Space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, space_sdk="docker", private=False)
    print(f"Space '{repo_id}' created.")

# Upload all deployment files to the Space
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",
)
print("Uploaded deployment files to the Hugging Face Space.")
