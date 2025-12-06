from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "aravindshaz3/tourism-package-predictor-2" # Changed repo_id for the Streamlit app
repo_type = "space"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id=repo_id,                                # the target repo
    repo_type="space",                              # dataset, model, or space
    path_in_repo=".",                               # Upload directly to the root of the space
)
