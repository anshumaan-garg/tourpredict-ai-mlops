from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "NineKnox/tourism-wellness-package"
repo_type = "dataset"

# Initialize API client with the write token stored in the HF_TOKEN environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset repo exists; create it if it does not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new dataset repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Step 2: Upload the local data folder to the Hugging Face dataset space
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Uploaded data folder to the Hugging Face dataset space.")
