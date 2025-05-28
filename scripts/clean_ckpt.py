import os
import shutil

# Base directory
base_dir = '/mnt/hdfs/tiktok_aiic/user/liuqian/verl_rl_checkpoints'
delete_folder = ["verl-grpo-fix-math-eval-large-reward"]

# Folder to keep
keep_folder = 'huggingface'

# Traverse the base directory
for root, dirs, files in os.walk(base_dir):
    # check if any allowed folders are in the current path
    if not any(folder in root for folder in delete_folder):
        print("Skipping this folder ", root)
        continue

    # Check if the current path matches the pattern
    if 'actor' in root and os.path.basename(root) == 'actor':
        for dir_name in dirs:
            if dir_name != keep_folder:
                # Remove unwanted directories use HDFS command
                hdfs_path = os.path.join(root, dir_name).replace("/mnt/hdfs/tiktok_aiic/", "hdfs://harunava/home/byte_data_seed_azureb_tteng/")
                print(f"Deleting {hdfs_path}")
                # use a subprocess to run the HDFS command
                os.system(f"hdfs dfs -rm -r {hdfs_path}")
    
# Output completion message
print("Cleanup complete. Only kept the 'huggingface' folder.")
