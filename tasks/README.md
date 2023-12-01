## Fine-Tuning Summarization Task

```
python finetune_summarization.py --model_id tiiuae/falcon-7b-instruct --wandb_name falcon-7b-instruct --use_model_prompt_defaults falcon --dataset beanham/medsum --input_col dialogue --target_col section_text --wandb_logging True --max_steps 250 --start_prompt 'Please summarize the following conversation:\n\n' --end_prompt '\n\nBegin summary:'
```

| Task                   | Llama-2-Chat | phi-1.5 | Mistral-7B | Vicuna | GPT-3.5T | GPT-4 |
| ---------------------- | ------------ | ------- | ---------- | ------ | -------- | ----- |
| FreshQA Fact-Checking  |              |         |            |        |          |       |
| CNN Text Summarization |              |         |            |        |          |       |
| Hybrid Hiring          |              |         |            |        |          |       |
