#!/bin/bash

# python3 zeroshot.py --model_id mistralai/Mistral-7B-Instruct-v0.1 --dataset yanyiwei/civil-comments-wilds --input_column text --target_column label --start_prompt '### Question: Does the following text contain toxicity? Respond with 1 (yes) or 0 (no) only. \n' --end_prompt '### Answer: '

# acc: 0.67
# f1: 0.7924528301886793
# wga: 0.5714285714285714

# python3 zeroshot.py --model_id mistralai/Mistral-7B-Instruct-v0.1 --dataset yanyiwei/civil-comments-wilds --input_column text --target_column label --start_prompt '###Example: Watch out for those white males. They are considered the enemy now a days. Answer: 1 (yes) ### Question: Does the following text contain toxicity? Respond with 1 (yes) or 0 (no) only. \n' --end_prompt '### Answer: '

# acc: 0.7
# f1: 0.8101265822784809
# wga: 0.3333333333333333

# python3 zeroshot.py --model_id mistralai/Mistral-7B-Instruct-v0.1 --dataset yanyiwei/civil-comments-wilds --input_column text --target_column label --start_prompt "###Examples: Watch out for those white males. They are considered the enemy now a days. Answer: 1 (yes)\nRussiaGate isn't over. Far from. But keeping bleating Trump Woolie. Baaaaaa! Answer: 0 (no)\n### Question: Does the following text contain toxicity? Respond with 1 (yes) or 0 (no) only. \n" --end_prompt '### Answer: '

# acc: 0.71
# f1: 0.8198757763975156
# wga: 0.3333333333333333

# python3 zeroshot.py --model_id mistralai/Mistral-7B-Instruct-v0.1 --dataset yanyiwei/civil-comments-wilds --input_column text --target_column label --start_prompt "###Examples: Watch out for those white males. They are considered the enemy now a days. Answer: 1 (yes)\nRussiaGate isn't over. Far from. But keeping bleating¡ Trump Woolie. Baaaaaa! Answer: 0 (no)\nBlack idiot or white idiot. What is the difference? Answer: 1 (yes)\n### Question: Does the following text contain toxicity? Respond with 1 (yes) or 0 (no) only. \n" --end_prompt '### Answer: '

# acc: 0.67
# f1: 0.7870967741935484
# wga: 0.3333333333333333

# CLOSED

# python3 fewshots.py --model_id mistralai/Mistral-7B-Instruct-v0.1 --dataset yanyiwei/civil-comments-sampled --nshot zero

# python3 fewshots.py --model_id meta-llama/Llama-2-7b-chat-hf --dataset yanyiwei/civil-comments-sampled --nshot three

# python3 fewshots.py --model_id tiiuae/falcon-7b-instruct --dataset yanyiwei/civil-comments-sampled --nshot three

# python3 finetune_summarization.py --dataset yanyiwei/civil-comments-wilds --model_id mistralai/Mistral-7B-Instruct-v0.1 --train_slice train --validation_slice validation --test_slice test

# python3 finetune_summarization.py --dataset yanyiwei/civil-comments-sampled --model_id mistralai/Mistral-7B-Instruct-v0.1 --train_slice train --validation_slice validation --test_slice test

# python3 finetune_summarization.py --dataset yanyiwei/civil-comments-sampled --model_id meta-llama/Llama-2-7b-chat-hf --train_slice train --validation_slice validation --test_slice test

python3 finetune_summarization.py --dataset yanyiwei/civil-comments-sampled --model_id tiiuae/falcon-7b-instruct --train_slice train --validation_slice validation --test_slice test

# OPEN

# python3 finetune_summarization_openai.py --model_id gpt-4 --OPENAI_API_KEY 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 --dataset yanyiwei/civil-comments-sampled

# python3 gpt_evaluation.py --dataset yanyiwei/civil-comments-sampled --shottype zero --key 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

# python3 gpt_evaluation.py --dataset yanyiwei/civil-comments-sampled --shottype one --key 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

# python3 gpt_evaluation.py --dataset yanyiwei/civil-comments-sampled --shottype two --key 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

# python3 gpt_evaluation.py --dataset yanyiwei/civil-comments-sampled --shottype three --key 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
