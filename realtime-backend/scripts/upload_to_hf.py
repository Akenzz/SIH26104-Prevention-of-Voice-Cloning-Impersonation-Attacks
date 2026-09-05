import os
from huggingface_hub import HfApi, create_repo

def upload_models():
    api = HfApi()
    repo_id = "Akenzz/SIH-Models"
    
    print(f"Ensuring repo {repo_id} exists...")
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    except Exception as e:
        print(f"Could not create repo (might already exist or permission issue): {e}")

    # 1. Upload WavLM model
    wavlm_path = "wavlm-base-plus/checkpoints/best_model_v5.pt"
    if os.path.exists(wavlm_path):
        print(f"Uploading {wavlm_path}...")
        api.upload_file(
            path_or_fileobj=wavlm_path,
            path_in_repo="wavlm_best_model_v5.pt",
            repo_id=repo_id,
            repo_type="model",
        )
        print("WavLM upload complete.")
    else:
        print(f"WavLM model not found at {wavlm_path}")

    # 2. Upload SSL model
    ssl_path = "ssl-detector/best_SSL_model_LA.pth"
    if os.path.exists(ssl_path):
        print(f"Uploading {ssl_path}...")
        api.upload_file(
            path_or_fileobj=ssl_path,
            path_in_repo="best_SSL_model_LA.pth",
            repo_id=repo_id,
            repo_type="model",
        )
        print("SSL upload complete.")
    else:
        print(f"SSL model not found at {ssl_path}")

if __name__ == "__main__":
    upload_models()
