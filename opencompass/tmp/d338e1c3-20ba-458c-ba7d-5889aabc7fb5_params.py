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
                            "Question:\n{prompt}\nAnswer:\nLet's think step by step.\n",
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
        abbr='huggingface_hf',
        batch_size=8,
        generation_kwargs=dict(),
        max_out_len=256,
        max_seq_len=None,
        model_kwargs=dict(),
        pad_token_id=None,
        path=
        '/cfs2/hadoop-aipnlp/zengweihao02/checkpoints/verl-grpo-Qwen2.5-0.5B-rollout-1024-256mini-remove-reward-tem1.0-8k-fix_abel_remove_level1to4_Qwen2.5-0.5B/global_step_100/actor/huggingface',
        peft_kwargs=dict(),
        peft_path=None,
        run_cfg=dict(num_gpus=1),
        stop_words=[],
        tokenizer_kwargs=dict(),
        tokenizer_path=None,
        type=
        'opencompass.models.huggingface_above_v4_33.HuggingFacewithChatTemplate'
    ),
]
work_dir = 'outputs/default/20250313_225537'
