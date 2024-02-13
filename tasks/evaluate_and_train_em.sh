######################### ZS HF MODELS ########################
## 0 shot
python evaluate_em.py --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 --run_name Mistral-7B-Instruct-v0.1_zs --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults mistral --model_type hf --wandb_name deeds-voting-match  --max_new_tokens 500
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-hf_zs --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --model_type hf --wandb_name deeds-voting-match  --max_new_tokens 500
python evaluate_em.py --hf_model_id tiiuae/falcon-7b-instruct --run_name falcon-7b-instruct_zs --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults falcon  --model_type hf --wandb_name deeds-voting-match  --max_new_tokens 500

# 1 shot
python evaluate_em.py --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 --run_name Mistral-7B-Instruct-v0.1_1s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults mistral --shots 1 --model_type hf --wandb_name deeds-voting-match  --max_new_tokens 500
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-hf_1s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 1 --model_type hf --wandb_name deeds-voting-match  --max_new_tokens 500
python evaluate_em.py --hf_model_id tiiuae/falcon-7b-instruct --run_name falcon-7b-instruct_1s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults falcon --shots 1 --model_type hf --wandb_name deeds-voting-match  --max_new_tokens 500

# 2 shot
python evaluate_em.py --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 --run_name Mistral-7B-Instruct-v0.1_2s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults mistral --shots 2 --model_type hf --wandb_name deeds-voting-match  --max_new_tokens 500
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-hf_2s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 2 --model_type hf --wandb_name deeds-voting-match  --max_new_tokens 500
python evaluate_em.py --hf_model_id tiiuae/falcon-7b-instruct --run_name falcon-7b-instruct_2s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults falcon --shots 2 --model_type hf --wandb_name deeds-voting-match  --max_new_tokens 500

# 3 shot
python evaluate_em.py --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 --run_name Mistral-7B-Instruct-v0.1_3s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults mistral --shots 3 --model_type hf --wandb_name deeds-voting-match  --max_new_tokens 500
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-hf_3s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 3 --model_type hf --wandb_name deeds-voting-match  --max_new_tokens 500
python evaluate_em.py --hf_model_id tiiuae/falcon-7b-instruct --run_name falcon-7b-instruct_3s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults falcon --shots 3 --model_type hf --wandb_name deeds-voting-match  --max_new_tokens 500


######################### FINETUNE HF MODELS ########################
python finetune_summarization.py --model_id tiiuae/falcon-7b-instruct --run_name falcon-7b-instruct_finetunedv2 --save_dir falcon-7b-instruct_deeds_voting_finetunedv2 --dataset isaacOnline/deeds-and-voting-matching  --input_col prompt --target_col overall_label --train_slice train --validation_slice validation --test_slice test --wandb_logging True --max_steps 700 --compute_em_metrics True --version --compute_summarization_metrics False --use_model_prompt_defaults falcon --hub_save_id isaacOnline/falcon-7b-instruct_finetunedv2 --peft_save_dir isaacOnline/falcon-7b-instruct_finetunedv2 --wandb_name deeds-voting-match
python finetune_summarization.py --model_id mistralai/Mistral-7B-Instruct-v0.1 --run_name Mistral-7B-Instruct-v0.1_finetunedv2 --save_dir Mistral-7B-Instruct-v0.1_deeds_voting_finetunedv2 --dataset isaacOnline/deeds-and-voting-matching  --input_col prompt --target_col overall_label --train_slice train --validation_slice validation --test_slice test --wandb_logging True --max_steps 700 --compute_em_metrics True --version --compute_summarization_metrics False --use_model_prompt_defaults mistral  --hub_save_id isaacOnline/Mistral-7B-Instruct-v0.1_finetunedv2 --peft_save_dir isaacOnline/Mistral-7B-Instruct-v0.1_finetunedv2 --wandb_name deeds-voting-match
python finetune_summarization.py --model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-hf_finetunedv3 --save_dir Llama-2-7b-chat-hf_deeds_voting_finetunedv2 --dataset isaacOnline/deeds-and-voting-matching  --input_col prompt --target_col overall_label --train_slice train --validation_slice validation --test_slice test --wandb_logging True --max_steps 700 --compute_em_metrics True --version --compute_summarization_metrics False --use_model_prompt_defaults llama-2 --hub_save_id isaacOnline/Llama-2-7b-chat-hf_finetunedv2 --peft_save_dir isaacOnline/Llama-2-7b-chat-hf_finetunedv2 --wandb_name deeds-voting-match


######################### ZS OPENAI MODELS ########################
## 0 shot
python evaluate_em.py --intermediate_outputs_dir gpt-3.5-turbo --oai_model_id gpt-3.5-turbo --run_name gpt-3.5-turbo --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults openai --model_type openai --shots 0 --wandb_name deeds-voting-match
python evaluate_em.py --intermediate_outputs_dir gpt-4 --oai_model_id gpt-4 --run_name gpt-4 --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults openai --model_type openai --shots 0 --wandb_name deeds-voting-match
python evaluate_em.py --intermediate_outputs_dir gpt-4-1106-preview --oai_model_id gpt-4-1106-preview --run_name gpt-4-1106-preview --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults openai --model_type openai --shots 0 --wandb_name deeds-voting-match

# 1 shot
python evaluate_em.py --intermediate_outputs_dir gpt-3.5-turbo --oai_model_id gpt-3.5-turbo --run_name gpt-3.5-turbo --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults openai --model_type openai --shots 1 --wandb_name deeds-voting-match
python evaluate_em.py --intermediate_outputs_dir gpt-4 --oai_model_id gpt-4 --run_name gpt-4 --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults openai --model_type openai --shots 1 --wandb_name deeds-voting-match
python evaluate_em.py --intermediate_outputs_dir gpt-4-1106-preview --oai_model_id gpt-4-1106-preview --run_name gpt-4-1106-preview --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults openai --model_type openai --shots 1 --wandb_name deeds-voting-match

# 2 shot
python evaluate_em.py --intermediate_outputs_dir gpt-3.5-turbo --oai_model_id gpt-3.5-turbo --run_name gpt-3.5-turbo --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults openai --model_type openai --shots 2 --wandb_name deeds-voting-match
python evaluate_em.py --intermediate_outputs_dir gpt-4 --oai_model_id gpt-4 --run_name gpt-4 --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults openai --model_type openai --shots 2 --wandb_name deeds-voting-match
python evaluate_em.py --intermediate_outputs_dir gpt-4-1106-preview --oai_model_id gpt-4-1106-preview --run_name gpt-4-1106-preview --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults openai --model_type openai --shots 2 --wandb_name deeds-voting-match

# 3 shot
python evaluate_em.py --intermediate_outputs_dir gpt-3.5-turbo --oai_model_id gpt-3.5-turbo --run_name gpt-3.5-turbo --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults openai --model_type openai --shots 3 --wandb_name deeds-voting-match
python evaluate_em.py --intermediate_outputs_dir gpt-4 --oai_model_id gpt-4 --run_name gpt-4 --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults openai --model_type openai --shots 3 --wandb_name deeds-voting-match
python evaluate_em.py --intermediate_outputs_dir gpt-4-1106-preview --oai_model_id gpt-4-1106-preview --run_name gpt-4-1106-preview --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults openai --model_type openai --shots 3 --wandb_name deeds-voting-match


######################## FINETUNE OPENAI MODELS ########################
# Finetune
python finetune_em_openai.py --intermediate_outputs_dir gpt-3.5-turbo-finetune --model_id gpt-3.5-turbo --run_name gpt-3.5-turbo-finetune --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults openai --wandb_name deeds-voting-match


######################### ZS CROSSOVER HF MODELS ########################
### 0 shot
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-hf_zs --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 0 --model_type hf_crossover --wandb_name deeds-voting-match  --max_new_tokens 500 --hf_lora_weights wolferobert3/llama-2-chat_factcheck_four_bit
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-hf_zs --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 0 --model_type hf_crossover --wandb_name deeds-voting-match  --max_new_tokens 500 --hf_lora_weights beanham/med-summarization-peft

## 1 shot
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-hf_1s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 1 --model_type hf_crossover --wandb_name deeds-voting-match  --max_new_tokens 500 --hf_lora_weights wolferobert3/llama-2-chat_factcheck_four_bit
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-hf_1s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 1 --model_type hf_crossover --wandb_name deeds-voting-match  --max_new_tokens 500 --hf_lora_weights beanham/med-summarization-peft

## 2 shot
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-hf_2s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 2 --model_type hf_crossover --wandb_name deeds-voting-match  --max_new_tokens 500 --hf_lora_weights wolferobert3/llama-2-chat_factcheck_four_bit
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-hf_2s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 2 --model_type hf_crossover --wandb_name deeds-voting-match  --max_new_tokens 500 --hf_lora_weights beanham/med-summarization-peft

## 3 shot
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-hf_3s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 3 --model_type hf_crossover --wandb_name deeds-voting-match  --max_new_tokens 500 --hf_lora_weights wolferobert3/llama-2-chat_factcheck_four_bit
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-hf_3s --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 3 --model_type hf_crossover --wandb_name deeds-voting-match  --max_new_tokens 500 --hf_lora_weights beanham/med-summarization-peft


######################### DOSE RESPONSE FOR LLAMA ########################
# 100%
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-dose --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 0 --model_type hf_crossover --wandb_name deeds-voting-match  --max_new_tokens 10 --hf_lora_weights isaacOnline/outputs_llama-2 --adapter_revision 24c22db9489c3d0b4a19a3534f568a8ea439a257

# 80%
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-dose --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 0 --model_type hf_crossover --wandb_name deeds-voting-match  --max_new_tokens 10 --hf_lora_weights isaacOnline/outputs_llama-2 --adapter_revision 627a3c88bab056356b12ed4a20a83a38caea6bdd

# 60%
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-dose --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 0 --model_type hf_crossover --wandb_name deeds-voting-match  --max_new_tokens 10 --hf_lora_weights isaacOnline/outputs_llama-2 --adapter_revision 3bac04b6a737dc6d5ba516fb24c61d1410b4c037

# 40%
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-dose --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 0 --model_type hf_crossover --wandb_name deeds-voting-match  --max_new_tokens 10 --hf_lora_weights isaacOnline/outputs_llama-2 --adapter_revision 056c64cc6639730aa4dd8c926ee142cb08c02646

# 20%
python evaluate_em.py --hf_model_id meta-llama/Llama-2-7b-chat-hf --run_name Llama-2-7b-chat-dose --dataset isaacOnline/deeds-and-voting-matching --input_col prompt --target_col overall_label --wandb_logging True --use_model_prompt_defaults llama-2 --shots 0 --model_type hf_crossover --wandb_name deeds-voting-match  --max_new_tokens 10 --hf_lora_weights isaacOnline/outputs_llama-2 --adapter_revision b2ee85129a01f6de8038aa8d564be1a5e57f1fa5



