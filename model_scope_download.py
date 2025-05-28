import os
# os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from modelscope import snapshot_download
models= ["bert123/R1-Verifier-1.5B","bert123/R1-Qwen-1.5B"]
# model="EleutherAI/llemma_34b"
# models=["bigcode/starcoder2-7b"]
for model in models:
    while True:
        try:
            snapshot_download(repo_id=model, ignore_patterns=[], local_dir=os.path.join("/Users/bytedance/Desktop/verl/model_scope_model/upload_model",model))
            break
        except KeyboardInterrupt:
            print("Program interrupted by the user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

        print("successful")