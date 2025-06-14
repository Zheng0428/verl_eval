# Copyright 2024 PRIME team and/or its affiliates
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

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from traceback import print_exc
from multiprocessing import cpu_count

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


async def single_compute_score(evaluation_func, completion, reference, task, extra_info, executor, timeout=300):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(evaluation_func, task, completion, reference, extra_info)  # Ensure synchronous
                ),
                timeout=timeout)
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion[:64]}...")
        return None  # Default value for timed-out rows
    except Exception:
        print_exc()
        return None  # Default value for failed rows


async def parallel_compute_score_async(evaluation_func,
                                       completions,
                                       references,
                                       tasks,
                                       extra_infos=None,
                                       num_processes=64):
    scores = []
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        # Create tasks for all rows
        tasks_async = [
            single_compute_score(evaluation_func,
                                 completion,
                                 reference,
                                 task,
                                 extra_info=extra_info,
                                 executor=executor,
                                 timeout=40)
            for completion, reference, task, extra_info in zip(completions, references, tasks, extra_infos)
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        results = await asyncio.gather(*tasks_async, return_exceptions=False)
        # try:
        #     results = await asyncio.gather(*tasks_async, return_exceptions=False)
        # except:
        #     # for pid, proc in executor._processes.items():
        #     #     try:
        #     #         proc.kill()
        #     #     except Exception as kill_err:
        #     #         print('shut down failed: ' + str(kill_err))
        #     raise

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores


class PrimeRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        sequences_strs = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truths = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        extra_infos = [data_item.non_tensor_batch.get('extra_info', None) for data_item in data]

        num_processes = max(cpu_count()-16, 1) # we keep 16 cpus in case they are being used by other processes

        print(f"Start to compute rewards for {len(sequences_strs)} samples over {num_processes} processes.")
        assert len(sequences_strs) == len(ground_truths) == len(data_sources)
        try:
            scores = asyncio.run(
                parallel_compute_score_async(self.compute_score,
                                             sequences_strs,
                                             ground_truths,
                                             data_sources,
                                             extra_infos=extra_infos,
                                             num_processes=num_processes))
        except asyncio.TimeoutError as e:
            print('Global timeout in reward computing! Setting all as 0.')
            scores = [0. for _ in range(len(sequences_strs))]
        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [0. for _ in range(len(sequences_strs))]

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                # print(sequences_str)

        return reward_tensor
