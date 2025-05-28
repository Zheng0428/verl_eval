import json

# data = json.load(open("genrm_train/data/skywork/train_data.json"))

data_files = [
    "/Users/bytedance/Desktop/verl/genrm_train/data/deepscaler_iter1/train_data.json",
    "/Users/bytedance/Desktop/verl/genrm_train/data/deepscaler/train_data.json"
]




data = []
for file in data_files:
    data.extend(json.load(open(file)))

print(len(data))
with open("genrm_train/data/deepscaler_iter0_iter1_merged_train_data.json", "w") as f:
    json.dump(data, f, indent=4)

