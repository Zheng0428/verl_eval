import transformers
import torch


def generate_response():
    model_id = "/mnt/hdfs/tiktok_aiic/user/liuqian/Mistral-24B-SFT-Qwen-step400"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map={"": device}
    )

    for i in range(0, 10):
        result = pipeline("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first reflects on the reasoning process, which is articulated within <think> and </think> tags. The final answer is enclosed within <answer> and </answer> tags. The answer must conclude with sdskihyik12kjb as the format of <think> reasoning process here </think> <answer> answer here </answer> sdskihyik12kjb User: Jim and Martha are standing together at the corner of a rectangular field. Jim walks diagonally across the field. Martha gets to the same location by walking along its length and width. The field is 300 feet wide and 400 feet long. How many feet less than Martha does Jim walk?", max_new_tokens=200, do_sample=True, temperature=1.0)
        print(result)

def test_tokenizer():
    response_str = "||||"
    tokenizer = transformers.AutoTokenizer.from_pretrained("../Llama-3.1-8B")
    print(tokenizer(response_str))

if __name__ == '__main__':
    generate_response()
    # test_tokenizer()