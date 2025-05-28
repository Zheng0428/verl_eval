import os
# os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from huggingface_hub import snapshot_download
models= ["yuzhen17/genrm_results"]
# model="EleutherAI/llemma_34b"
# models=["bigcode/starcoder2-7b"]
for model in models:
    while True:
        try:
            snapshot_download(repo_id=model,max_workers=128,resume_download=True,ignore_patterns=[],local_dir_use_symlinks=False,token="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", local_dir="genrm_results")
            break
        except KeyboardInterrupt:
            print("Program interrupted by the user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

        print("successful")