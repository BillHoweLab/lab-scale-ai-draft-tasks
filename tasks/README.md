## Fine-Tuning Summarization Task

```
python finetune_summarization.py --model_id tiiuae/falcon-7b-instruct --wandb_name falcon-7b-instruct --use_model_prompt_defaults falcon --dataset beanham/medsum --input_col dialogue --target_col section_text --wandb_logging True --max_steps 500 --eval_steps 25 --save_steps 25 --start_prompt 'Please summarize the following conversation:\n\n' --end_prompt '\n\nBegin summary:'
```

```
python pretrain_evaluation.py --model_id tiiuae/falcon-7b-instruct --use_model_prompt_defaults falcon --shot 1 --dataset beanham/medsum --input_col dialogue --target_col section_text --start_prompt 'Please summarize the following conversation:\n\n' --end_prompt '\n\nBegin summary:'
```

| Task                   | Llama-2-Chat | Mistral-7B | Falcon | GPT-3.5T | GPT-4 |
| ---------------------- | ------------ | ---------- | ------ | -------- | ----- |
| Clinical Summarization |              |            |        |          |       |
