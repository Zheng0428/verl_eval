from math_verify import parse
import pandas as pd
from math_verify import parse, verify
data_list = [
    "/Users/bytedance/Desktop/verl/examples/data_preprocess/converted_data/simplelr_skywork/train.parquet",
    "/Users/bytedance/Desktop/verl/examples/data_preprocess/converted_data/simplelr_orz/train.parquet",
    "/Users/bytedance/Desktop/verl/examples/data_preprocess/converted_data/simplelr_math_35/train.parquet"
]

# hf_can_parse = []
# for data_path in data_list:
#     print(data_path)
#     data = pd.read_parquet(data_path)
#     for idx, row in data.iterrows():
#         ans= row['reward_model']['ground_truth']
#         # try:
#         if "\\boxed" not in ans:
#             ans = f"\\boxed{{{ans}}}"
#         a = parse(ans)
#         # print(a)
#         if a != []:
#             hf_can_parse.append(ans)
#         else:
#             print(ans)
#         # except:
#         #     pass
#     print("can parse: ", len(hf_can_parse))
#     print("can not parse: ", len(data) - len(hf_can_parse))
#     print("total: ", len(data))
#     print("--------------------------------")
# print(len(hf_can_parse))

case_list = [
    {
        "gold": "\\frac{82944}{456375}",
        "pred": "0.1817"
    },
    {
        "gold": "\\frac{4 \\sqrt{21}}{9}",
        "pred": "2.03670030"
    },
    {
        "gold": "\\frac{1}{4}\\cdot\\frac{1}{(\\cosx+\\sinx)^{4}}+C",
        "pred": "\\frac{1}{4}\\cdot\\frac{1}{(\\cosx+\\sinx)^{4}} + N"
    }
]
for case in case_list:
    if "\\boxed" not in case["gold"]:
        case["gold"] = f"\\boxed{{{case['gold']}}}"
    if "\\boxed" not in case["pred"]:
        case["pred"] = f"\\boxed{{{case['pred']}}}"
    gold = parse(case["gold"])
    pred = parse(case["pred"])
    print(gold)
    print(pred)
    print(verify(gold, pred, numeric_precision=1))
    print("--------------------------------")
    
