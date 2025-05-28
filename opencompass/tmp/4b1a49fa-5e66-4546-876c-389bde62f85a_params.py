datasets = [
    [
        dict(
            abbr='GPQA_diamond_4',
            eval_cfg=dict(
                evaluator=dict(type='opencompass.datasets.GPQAEvaluator'),
                pred_postprocessor=dict(
                    type='opencompass.datasets.GPQA_Simple_Eval_postprocess')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.GenInferencer'),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt=
                            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAnswer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant",
                            role='HUMAN'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            name='gpqa_diamond.csv',
            path='./data/gpqa/',
            reader_cfg=dict(
                input_columns=[
                    'question',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='answer',
                test_range='[160:200]'),
            type='opencompass.datasets.GPQADataset'),
    ],
]
models = [
    dict(
        abbr='huggingface_hf-vllm',
        batch_size=8,
        generation_kwargs=dict(stop_token_ids=None),
        max_out_len=8192,
        max_seq_len=None,
        model_kwargs=dict(max_model_len=None, tensor_parallel_size=1),
        path=
        '/cfs2/hadoop-aipnlp/zengweihao02/checkpoints/verl-grpo-deepseek-math-base-rollout-1024-256mini-remove-reward-tem1.0-fix_qwen_remove_gsm8k_level1_deepseek-math-7b-base/global_step_100/actor/huggingface',
        run_cfg=dict(num_gpus=1),
        type='opencompass.models.vllm.VLLM'),
]
work_dir = '/cfs2/hadoop-aipnlp/zengweihao02/checkpoints/verl-grpo-deepseek-math-base-rollout-1024-256mini-remove-reward-tem1.0-fix_qwen_remove_gsm8k_level1_deepseek-math-7b-base/global_step_100/actor/huggingface/gpqa_qwen_box/20250314_220806'
