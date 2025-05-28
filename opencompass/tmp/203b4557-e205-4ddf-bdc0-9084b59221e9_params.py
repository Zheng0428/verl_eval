datasets = [
    [
        dict(
            abbr='IFEval_1',
            eval_cfg=dict(
                evaluator=dict(type='opencompass.datasets.IFEvaluator'),
                pred_role='BOT'),
            infer_cfg=dict(
                inferencer=dict(
                    max_out_len=1025,
                    type='opencompass.openicl.icl_inferencer.GenInferencer'),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt=
                            '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n',
                            role='HUMAN'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='data/ifeval/input_data.jsonl',
            reader_cfg=dict(
                input_columns=[
                    'prompt',
                ],
                output_column='reference',
                test_range='[271:542]'),
            type='opencompass.datasets.IFEvalDataset'),
    ],
]
models = [
    dict(
        abbr='huggingface_hf-vllm',
        batch_size=16,
        max_out_len=8192,
        max_seq_len=None,
        model_kwargs=dict(max_model_len=None, tensor_parallel_size=1),
        path=
        '/cfs2/hadoop-aipnlp/zengweihao02/checkpoints/verl-grpo-deepseek-math-base-rollout-1024-256mini-remove-reward-tem1.0-fix_qwen_remove_gsm8k_level1_deepseek-math-7b-base/global_step_100/actor/huggingface',
        run_cfg=dict(num_gpus=1),
        stop_words=[],
        type='opencompass.models.vllm_with_tf_above_v4_33.VLLMwithChatTemplate'
    ),
]
work_dir = '/cfs2/hadoop-aipnlp/zengweihao02/checkpoints/verl-grpo-deepseek-math-base-rollout-1024-256mini-remove-reward-tem1.0-fix_qwen_remove_gsm8k_level1_deepseek-math-7b-base/global_step_100/actor/huggingface/IFEval_gen_qwen_box/20250314_220628'
