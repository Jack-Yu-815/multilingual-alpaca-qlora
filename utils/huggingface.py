from huggingface_hub import HfApi

def upload_model_to_hf(folder_path: str, repo_id: str):
    api = HfApi()
    # Create the repository if it doesn't exist
    repo = api.create_repo(repo_id, repo_type="model")
    # Upload the folder to the specified repository
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo.repo_id,
        repo_type=repo.repo_type,
    )