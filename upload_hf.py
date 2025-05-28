import os
# os.environ["HF_T"]="https://hf-mirror.com" # No need now
import argparse
from huggingface_hub import HfApi, create_repo
# os.environ["HF_ENDPOINT"]="https://hf-mirror.com" # No need now
import huggingface_hub

hf_token="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"
parser = argparse.ArgumentParser()

parser.add_argument("--username", type=str, required=True, help="Your Hugging Face username")
parser.add_argument("--repo_id", type=str, required=True, help="The repository ID on Hugging Face")
parser.add_argument("--model_data_path", type=str, required=True, help="The local model folder path")
parser.add_argument("--repo_type", type=str, default="model", help="The type of data to upload (model, dataset, space)", choices=["model", "dataset", "space"])
parser.add_argument("--ignore_patterns", type=str, default="", help="Comma-separated list of patterns to ignore when uploading files")
parser.add_argument("--path_in_repo", type=str, default=None, help="The path in the repository")
parser.add_argument("--public", action="store_true", help="Whether to create a public repository")

args = parser.parse_args()
# 你的 Hugging Face 用户名
username = args.username
# 你想要推送的模型仓库名称
repo_id = args.repo_id
# 本地模型文件夹路径
model_path = args.model_data_path

# 初始化 Hugging Face API
api = HfApi()

# 检查仓库是否存在
try:
    # Ensure repo_id includes username if not already
    if "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"
        
    repo_info = api.repo_info(repo_id=repo_id, repo_type=args.repo_type)
    print(f"Repository '{repo_id}' already exists.")
except Exception as e:
    print(f"Repository '{repo_id}' does not exist or error occurred: {str(e)}")
    try:
        # Ensure repo_id includes username if not already
        if "/" not in repo_id:
            repo_id = f"{username}/{repo_id}"
            
        create_repo(repo_id=repo_id, private=not args.public, repo_type=args.repo_type, token=hf_token)
        print(f"Successfully created repository '{repo_id}'.")
    except huggingface_hub.errors.HfHubHTTPError as e:
        if "409" in str(e):
            print(f"Repository '{repo_id}' already exists but couldn't be accessed. Continuing with upload.")
        else:
            raise e

ignore_patterns = args.ignore_patterns.split(",") if args.ignore_patterns else None

if args.path_in_repo is None:
    path_in_repo=args.model_data_path.replace("/mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/", "")
else:
    path_in_repo=args.path_in_repo
# 上传整个文件夹到 Hugging Face Hub
api.upload_folder(
    folder_path=args.model_data_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type=args.repo_type,  # 你可以选择 'dataset' 或 'space'，视情况而定
    commit_message="Initial commit",
    ignore_patterns=ignore_patterns,
    token=hf_token
)

print("Files have been pushed to Hugging Face Hub.")