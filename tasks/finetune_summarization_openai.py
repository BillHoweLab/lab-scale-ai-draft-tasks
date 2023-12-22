#!/usr/bin/env python3

import json
import numpy as np
import torch
import bitsandbytes as bnb
import logging
import sys
import transformers
import datasets
import argparse
import wandb

from transformers import TrainingArguments
from huggingface_hub import login as hf_login
from os import path, mkdir, getenv
from typing import Mapping

from finetune_functions import format_data_as_instructions, get_model_and_tokenizer, get_lora_model, get_default_trainer, get_dataset_slices
from evaluate_functions import evaluate_hf_model

import openai
from openai import OpenAI
from openai_chat_api import DialogueBot

MODEL_SUFFIXES = {
    'openai': '',
    'mistral': '</s>',
    'llama-2': '</s>',
    'falcon': '<|endoftext|>',
    'opt-finetune': '</s>',
}

def format_for_finetuning(user_input: str,
                       assistant_output: str,
                       system_prompt: str='You are a helpful assistant specializing in fact-checking.') -> str:
    """
    Format data in JSON for fine-tuning an OpenAI chatbot model.
    """

    return json.dumps({"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}, {"role": "assistant", "content": assistant_output}]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune a summarization model.')

    # Model ID
    parser.add_argument('--model_id', type=str, default='facebook/opt-125m', help='The model ID to fine-tune.')
    parser.add_argument('--hf_token_var', type=str, default='HF_TOKEN', help='Name of the HuggingFace API token variable name.')
    parser.add_argument('--resume_from_checkpoint', type=str, default='False', help='Whether to resume from a checkpoint.')

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to mount the model on.')
    parser.add_argument('--use_mps_device', type=str, default='False', help='Whether to use an MPS device.')
    parser.add_argument('--max_memory', type=str, default='12000MB', help='The maximum memory per GPU, in MB.')

    # Model arguments
    parser.add_argument('--gradient_checkpointing', type=str, default='True', help='Whether to use gradient checkpointing.')
    parser.add_argument('--quantization_type', type=str, default='4bit', help='The quantization type to use for fine-tuning.')
    parser.add_argument('--lora', type=str, default='True', help='Whether to use LoRA.')
    parser.add_argument('--tune_modules', type=str, default='linear4bit', help='The modules to tune using LoRA.')
    parser.add_argument('--exclude_names', type=str, default='lm_head', help='The names of the modules to exclude from tuning.')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cnn_dailymail', help='The dataset to use for fine-tuning.')
    parser.add_argument('--version', type=str, default='3.0.0', nargs='?', help='The version of the dataset to use for fine-tuning.')
    parser.add_argument('--input_col', type=str, default='article', help='The name of the input column in the dataset.')
    parser.add_argument('--target_col', type=str, default='highlights', help='The name of the target column in the dataset.')
    parser.add_argument('--train_slice', type=str, default='train', help='The slice of the training dataset to use for fine-tuning.')
    parser.add_argument('--validation_slice', type=str, default='validation', help='The slice of the validation dataset to use for fine-tuning.')
    parser.add_argument('--test_slice', type=str, default='test', help='The slice of the test dataset to use for fine-tuning.')

    # Saving arguments
    parser.add_argument('--save_model', type=str, default='True', help='Whether to save the fine-tuned model and tokenizer.')
    parser.add_argument('--results_dir', type=str, default='finetuned_model', help='The directory to save the fine-tuned model and tokenizer.')
    parser.add_argument('--formatted_data_dir', type=str, help='The directory to save the formatted data to', default='formatted_data')
    parser.add_argument('--intermediate_outputs_dir', type=str, help='The directory to save intermediate outputs to', default='intermediate_outputs')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='logs', help='The directory to save the log file.')
    parser.add_argument('--log_level', type=str, default='info', help='The log level to use for fine-tuning.')
    parser.add_argument('--logging_first_step', type=str, default='False', help='Whether to log the first step.')
    parser.add_argument('--logging_steps', type=int, default=1, help='The number of steps between logging.')
    parser.add_argument('--run_name', type=str, default='peft_finetune', help='The name of the run, for logging.')

    # W&B logging arguments
    parser.add_argument('--wandb_logging', type=str, default='True', help='Whether to log to W&B.')
    parser.add_argument('--wandb_name', type=str, default='peft_finetune', help='The name of the W&B project, for logging.')
    parser.add_argument('--wandb_api_var', type=str, default='WANDB_API_KEY', help='Name of the WandB API key variable name.')

    # Prompt arguments
    parser.add_argument('--start_prompt', type=str, default='Please summarize the following conversation:\n\n', help='The start prompt to add to the beginning of the input text.')
    parser.add_argument('--end_prompt', type=str, default='\n\nBegin summary:', help='The end prompt to add to the end of the input text.')
    parser.add_argument('--suffix', type=str, default='</s>', help='The suffix to add to the end of the input and target text.')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='The maximum sequence length to use for fine-tuning.')
    parser.add_argument('--use_model_prompt_defaults', type=str, default='mistral', help='Whether to use the default prompts for a model')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use for fine-tuning.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='The number of gradient accumulation steps to use for fine-tuning.')
    parser.add_argument('--warmup_steps', type=int, default=10, help='The number of warmup steps to use for fine-tuning.')
    parser.add_argument('--max_steps', type=int, default=-1, help='The maximum number of steps to use for fine-tuning.')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='The learning rate to use for fine-tuning.')
    parser.add_argument('--fp16', type=str, default='True', help='Whether to use fp16.')
    parser.add_argument('--optim', type=str, default='paged_adamw_8bit', help='The optimizer to use for fine-tuning.')

    # Evaluation arguments
    parser.add_argument('--evaluation_strategy', type=str, default='steps', help='The evaluation strategy to use for fine-tuning.')
    parser.add_argument('--eval_steps', type=int, default=10, help='The number of steps between evaluations.')
    parser.add_argument('--eval_on_test', type=str, default='True', help='Whether to evaluate the model on the test set after fine-tuning.')
    parser.add_argument('--compute_summarization_metrics', type=str, default='True', help='Whether to evaluate the model on ROUGE, BLEU, and BERTScore after fine-tuning.')
    parser.add_argument('--compute_qanda_metrics', type=str, default='False', help='Whether to evaluate the model on QA metrics like F1 and Exact Match (from SQUAD).')
    parser.add_argument('--compute_em_metrics', type=str, default='False', help='Whether to evaluate the model on Accuracy, Precision, Recall, and F1.')

    # Hub arguments
    parser.add_argument('--hub_upload', type=str, default='False', help='Whether to upload the model to the hub.')
    parser.add_argument('--hub_save_id', type=str, default='wolferobert3/opt-125m-peft-summarization', help='The name under which the mode will be saved on the hub.')
    parser.add_argument('--save_steps', type=int, default=10, help='The number of steps between saving the model to the hub.')

    # Parse arguments
    args = parser.parse_args()

    # change saving directory
    args.save_dir = 'finetuned_model_openai'
    args.log_dir = 'logs_openai'
    args.output_dir = 'outputs_openai'
    args.run_name = 'peft_model_openai'
    
    # Update the start and end prompts if using the model defaults
    if args.use_model_prompt_defaults:
        args.suffix = MODEL_SUFFIXES[args.use_model_prompt_defaults]
        
    # Initialize W&B
    if args.wandb_logging == 'True':
        wandb.login(key=getenv(args.wandb_api_var))
        wandb.init(project=args.wandb_name, 
                   name=args.run_name, 
                   config=args)
    
    # Create directories if they do not exist
    if not path.exists(args.results_dir):
        mkdir(args.results_dir)
        print(f'Created directory {args.results_dir}')
    
    if not path.exists(args.log_dir):
        mkdir(args.log_dir)
        print(f'Created directory {args.log_dir}')

    if not path.exists(args.formatted_data_dir):
        mkdir(args.formatted_data_dir)
        print(f'Created directory {args.formatted_data_dir}')

    if not path.exists(args.intermediate_outputs_dir):
        mkdir(args.intermediate_outputs_dir)
        print(f'Created directory {args.intermediate_outputs_dir}')
        
    # Create a logger
    logger = logging.getLogger(__name__)

    # Setup logging
    print('Setting up logging...')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Use the default log level matching the training args
    log_level = args.log_level.upper()
    logger.setLevel(log_level)

    # Set the log level for the transformers and datasets libraries
    transformers.utils.logging.get_logger("transformers").setLevel(log_level)
    datasets.utils.logging.get_logger("datasets").setLevel(log_level)

    # Log to file
    file_handler = logging.FileHandler(path.join(args.log_dir, f'{args.run_name}.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(file_handler)

    # ==================
    # chat format 
    # ==================
    system_message = """You are a helpful medical assistant! Please help me summarize dialogues between doctors and patients. I will provide you with the content and topic for each dialogue. """
    transaction = """\n\nPlease summarize the following dialogue."""
    
    # Download and prepare data
    print('Downloading and preparing data...')

    data = get_dataset_slices(args.dataset,
                              args.version,
                              train_slice=args.train_slice,
                              validation_slice=args.validation_slice,
                              test_slice=args.test_slice)

    # Set the format of the data
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']

    # Format data for fine-tuning
    print('Formatting data for fine-tuning...')
    train_data_formatted = '\n'.join([format_for_finetuning(train_data[args.input_col][i], train_data[args.target_col][i], args.system_prompt) for i in range(len(train_data))])
    validation_data_formatted = '\n'.join([format_for_finetuning(validation_data[args.input_col][i], validation_data[args.target_col][i], args.system_prompt) for i in range(len(validation_data))])
    
    # Instantiate trainer
    print('Instantiating trainer...')

    training_args = TrainingArguments(
            per_device_train_batch_size=args.batch_size, ## 1
            gradient_accumulation_steps=args.gradient_accumulation_steps, ## 4
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate,
            fp16=args.fp16 == 'True',
            output_dir=args.output_dir,
            optim=args.optim,
            use_mps_device=args.use_mps_device == 'True',
            log_level=args.log_level,
            evaluation_strategy=args.evaluation_strategy,            
            resume_from_checkpoint=args.resume_from_checkpoint == 'True',
            push_to_hub=args.hub_upload == 'True',            
            report_to=['wandb'] if args.wandb_logging == 'True' else [],
            max_steps=args.max_steps, ## -1
            logging_first_step=args.logging_first_step == 'True',
            logging_steps=args.logging_steps, ## 30 ~ 0.1 epoch ~ 10% of training data
            eval_steps=args.eval_steps, ## 30 ~ 0.1 epoch ~ 10% of training data
            save_steps=args.save_steps, ## 30 ~ 0.1 epoch ~ 10% of training data
            load_best_model_at_end=True,
            num_train_epochs=1,
        )

    trainer = get_default_trainer(model, 
                                  tokenizer, 
                                  data['train'], 
                                  eval_dataset=data['validation'],
                                  formatting_func=data_formatter,
                                  max_seq_length=args.max_seq_length,
                                  training_args=training_args)
    
    model.config.use_cache = False

    logger.info(f'Instantiated Trainer')

    # Fine-tune model
    print('Fine-tuning model...')

    trainer.train()

    logger.info(f'Completed fine-tuning')

    # Save adapter weights and tokenizer
    if args.save_model == 'True':

        print('Saving model and tokenizer...')

        trainer.model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)

        logger.info(f'Saved model and tokenizer to {args.save_dir}')

    # Save model to hub
    if args.hub_upload == 'True':

        print('Saving model to hub...')

        trainer.model.push_to_hub(args.hub_save_id, use_auth_token=True)

        logger.info(f'Saved model to hub')
        
    if args.wandb_logging == 'True':
        wandb.finish()
