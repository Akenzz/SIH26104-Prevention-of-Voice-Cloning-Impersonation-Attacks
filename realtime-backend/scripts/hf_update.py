import os
from huggingface_hub import HfApi

def update_hf():
    api = HfApi()
    repo_id = "Akenzz/SIH-Models"
    
    # Delete old models
    for file_to_delete in ["wavlm_best_model_v5.pt", "best_SSL_model_LA.pth"]:
        try:
            print(f"Deleting {file_to_delete}...")
            api.delete_file(path_in_repo=file_to_delete, repo_id=repo_id, repo_type="model")
            print(f"Deleted {file_to_delete}.")
        except Exception as e:
            print(f"Failed to delete {file_to_delete}: {e}")

    # Upload new model
    new_model_path = "/home/akenzz/sih/project/wavlm-base-plus/checkpoints/best_model_v6.pt"
    if os.path.exists(new_model_path):
        print(f"Uploading {new_model_path} as best_model_v6.pt...")
        api.upload_file(
            path_or_fileobj=new_model_path,
            path_in_repo="best_model_v6.pt",
            repo_id=repo_id,
            repo_type="model",
        )
        print("Upload complete.")
    else:
        print(f"New model not found at {new_model_path}")

if __name__ == "__main__":
    update_hf()
