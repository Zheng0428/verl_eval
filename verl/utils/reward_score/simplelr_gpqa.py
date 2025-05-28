# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import re
from .qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
# from .qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal
from functools import partial
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import threading
import logging
from typing import Optional, Callable, Any
from functools import wraps
import random
import gc 
from math_verify import parse, verify

class GlobalProcessPool:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, max_workers: int = 16, reset_threshold: int = 100000):
        self.max_workers = max_workers
        self.reset_threshold = reset_threshold
        self.task_counter = 0
        self.executor: Optional[ProcessPoolExecutor] = None
        self.logger = logging.getLogger(__name__)
        self._initialize_executor()
    
    def _initialize_executor(self) -> None:
        """Initialize a new ProcessPoolExecutor and reset task counter."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None
            gc.collect() 
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.task_counter = 0
        self.logger.warning(f"Initialized ProcessPoolExecutor with {self.max_workers} workers")
    
    @classmethod
    def get_instance(cls, max_workers: int = 16, reset_threshold: int = 100000) -> 'GlobalProcessPool':
        """Get or create the singleton instance of GlobalProcessPool."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_workers=max_workers, reset_threshold=reset_threshold)
        return cls._instance
    
    def submit(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Submit a task to the executor with automatic recovery and periodic reset.
        
        Args:
            fn: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Future object representing the computation
        """
        try:
            if self.executor is None:
                with self._lock:
                    self._initialize_executor()
            return self.executor.submit(fn, *args, **kwargs)
        except (Exception, RuntimeError) as e:
            self.logger.warning(f"Process pool broken, recreating: {str(e)}")
            with self._lock:
                self._initialize_executor()
            return self.executor.submit(fn, *args, **kwargs)

# Create the global executor instance
global_executor = GlobalProcessPool.get_instance(max_workers=16)

def extract_last_boxed(text):
    """
    提取 LaTeX 文本中最后一个 \boxed 命令中的内容
    
    返回:
    - str: 最后一个 \boxed 中的内容。如果没有找到则返回 None
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    
    # 找到所有匹配
    matches = list(re.finditer(pattern, text))
    
    # 如果找到匹配，返回最后一个的内容
    if matches:
        return matches[-1].group(0)
    return None

    
def extract_solution(solution_str):
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    
    predict_answer = qwen_extract_answer(model_output, data_name="math")
    extract_boxed_answer = extract_last_boxed(model_output)
    # True means the boxed answer is correct
    if extract_boxed_answer is not None:
        return predict_answer, True
    else:
        return predict_answer, False


def clean_solution(solution_str):
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    return model_output

def first_option_postprocess(text: str, options: str, cushion=False) -> str:
    """Find first valid option for text."""
    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        # 修复LaTeX boxed正则表达式（添加大括号转义）
        f'\\\\boxed{{\\s*([{options}])\\s*}}',
        f'\\\\boxed{{([{options}])}}',
        f'boxed{{([{options}])}}',
        f'[Oo]ption\s+([{options}])\s+is\s+correct',
        f'[Oo]ption\\s+([{options}])\\s+is\\s+correct[.?!,]?',
        f'(?i)answer\\s*:\\s*([{options}])[.。?!,]?',
        f'答案是?\s*([{options}])',
        f'答案是?\s*：\s*([{options}])',
        f'答案是?\s*:\s*([{options}])',
        f'答案选项应?该?是\s*([{options}])',
        f'答案选项应?该?为\s*([{options}])',
        f'答案应该?是\s*([{options}])',
        f'答案应该?选\s*([{options}])',
        f'答案选项为?\s*：\s*([{options}])',
        f'答案选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'答案选项是?\s*:\s*([{options}])',
        f'答案为\s*([{options}])',
        f'答案选\s*([{options}])',
        f'选择?\s*([{options}])',
        f'故选?\s*([{options}])'
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'(?i)ANSWER\s*:\s*([{options}])',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        f'^选项\s?([{options}])',
        f'^([{options}])\s?选?项',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        f'1.\s?(.*?)$',
        f'1.\s?([{options}])[.。$]?$',
    ]
    # cushion_patterns = [
    #     f'([{options}]):',
    #     f'([{options}])',
    # ]
    # flake8: noqa
    # yapf: enable

    # if cushion:
    #     patterns.extend(cushion_patterns)
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if match.group(1) is not None and match.group(1) != '':
                outputs = match.group(1)
            else:
                outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''


def gpqa_verify_with_try(gold, target):
    try:
        parsed_target = first_option_postprocess(target, options='ABCD')
        
        if type(gold) == str:
            gold = [gold]
        for g in gold:
            if g == parsed_target:
                return True
        return False
    except Exception as e:
        print(f"Gold: {gold} Target: {target} Error: {str(e)}")
        return False


def gpqa_equal_subprocess(gold, target, timeout_seconds=10):
    """
    使用 ProcessPoolExecutor 实现带超时的函数执行
    
    Args:
        gold: 参考答案
        target: 预测结果
        timeout_seconds: 超时时间(秒)
        
    Returns:
        bool: 执行结果,超时返回 False
    """
    try:
        # 提交任务到进程池
        future = global_executor.submit(gpqa_verify_with_try, gold=gold, target=target)
        # 等待结果,支持超时
        result = future.result(timeout=timeout_seconds)
        return result
    except TimeoutError:
        print(f"Timeout occurred for gold {gold} and target {target}.")
        return False
    except Exception as e:
        print(f"Gold: {gold} Target: {target} Error: {str(e)}")
        return False


# import os 
# # TODO: Might have problem in multi node ray cluster !!!!
# reward_function_type = str(os.environ.get('REWORD_FUNCTION_TYPE', "mix"))
# format_penalty_value = float(os.environ.get('FORMAT_PENALTY_VALUE', "-1"))

# print(f"Reward function type: {reward_function_type}")
# print(f"Format penalty value: {format_penalty_value}")

def compute_score(solution_str, ground_truth, method='strict', options='ABCD'):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    clean_solution_str = clean_solution(solution_str)
    parsed_target = first_option_postprocess(clean_solution_str, options=options)
    
    is_boxed_matched = parsed_target != ''
    if type(ground_truth) == str:
        ground_truth = [ground_truth]
    for g in ground_truth:
        if g.strip().lower() == parsed_target.strip().lower():
            correct = True
            break
        else:
            correct = False
    if correct:
        box_match = 1.0
    else:
        box_match = 0

    if random.random() < 0.05:
        # for 5% of the cases, print; otherwise, print nothing to accelerate the process 
        print(f"\n[Model Response]\n{solution_str}")
        print(f"\n[Ground Truth]\n{ground_truth}")
        print(f"\n[Is Boxed Matched]\n{is_boxed_matched}")
        print(f"\n[Extracted Answer]\n{parsed_target}")
        print(f"\n[Reward Score]\n{box_match}")
    return {"score": box_match, "extra_info": { "score": box_match}}



# def compute_accuracy(solution_str, ground_truth, method='strict'):
#     """The scoring function for GSM8k.

#     Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

#     Args:
#         solution_str: the solution text
#         ground_truth: the ground truth
#         method: the method to extract the solution, choices are 'strict' and 'flexible'
#         format_score: the score for the format
#         score: the score for the correct answer
#     """
#     extract_answer, is_boxed_matched = extract_solution(solution_str=solution_str)
    
    
#     if "\\boxed" not in extract_answer:
#         boxed_answer = f"\\boxed{{{extract_answer}}}"
#     else:
#         boxed_answer = extract_answer
    
#     if "\\boxed" not in ground_truth:
#         boxed_ground_truth = f"\\boxed{{{ground_truth}}}"
#     else:
#         boxed_ground_truth = ground_truth
        
    
#     target = parse(boxed_answer)    
#     gold = parse(boxed_ground_truth)
#     correct = hf_math_equal_subprocess(gold=gold, target=target)
    
    
#     if random.random() < 0.05:
#         # for 5% of the cases, print; otherwise, print nothing to accelerate the process 
#         print(f"\n[Model Response]\n{solution_str}")
#         print(f"\n[Ground Truth]\n{ground_truth}")
#         print(f"\n[Is Boxed Matched]\n{is_boxed_matched}")
#         print(f"\n[Extracted Answer]\n{extract_answer}")
#         print(f"\n[Reward Score]\n{correct}")
#     return float(correct)




if __name__ == "__main__":
    solution_str = """Answer the following multiple choice question. The last line of your response should be within \boxed{LETTER} where LETTER is one of ABCDEFGHIJ. Think step by step before answering.assistant<|endoftext|>"""
    model_output= first_option_postprocess(solution_str, options='ABCD')
    compute_score(solution_str, 'A', options='ABCD')
    print(model_output)