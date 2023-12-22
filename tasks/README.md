## Fine-Tuning Summarization Task

```
python finetune_summarization.py --model_id tiiuae/falcon-7b-instruct --wandb_name falcon-7b-instruct --use_model_prompt_defaults falcon --dataset beanham/medsum --eval_steps 30 --save_steps 30 --logging_steps 30
```

```
python finetune_summarization.py --model_id gpt3.5-turbo --wandb_name gpt3.5-turbo --use_model_prompt_defaults openai --dataset beanham/medsum
```

```
python evaluate_summarization.py --model_id tiiuae/falcon-7b-instruct --use_model_prompt_defaults falcon --dataset beanham/medsum --nshot zero --pretrain True
```

| Task                   | Llama-2-Chat | Mistral-7B | Falcon | GPT-3.5T | GPT-4 |
| ---------------------- | ------------ | ---------- | ------ | -------- | ----- |
| Clinical Summarization |              |            |        |          |       |
