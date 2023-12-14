## Fine-Tuning Summarization Task

```
python finetune_summarization.py --model_id tiiuae/falcon-7b-instruct --wandb_name falcon-7b-instruct --use_model_prompt_defaults falcon --dataset beanham/medsum --wandb_logging True --max_steps 500 --eval_steps 25 --save_steps 25
```

```
python evaluate_summarization.py --model_id tiiuae/falcon-7b-instruct --use_model_prompt_defaults falcon --dataset beanham/medsum --nshot zero --pretrain True
```

| Task                   | Llama-2-Chat | Mistral-7B | Falcon | GPT-3.5T | GPT-4 |
| ---------------------- | ------------ | ---------- | ------ | -------- | ----- |
| Clinical Summarization |              |            |        |          |       |
