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

    # Parse arguments
    args = parser.parse_args()

    # change saving directory
    args.results_dir = 'finetuned_model_openai'
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
    system_message = """You are a helpful medical assistant! Please help me summarize dialogues between doctors and patients. I will provide you with the content and topic for each dialogue. """
    transaction = """\n\nPlease summarize the following dialogue."""

    # Format data for fine-tuning
    print('Formatting data for fine-tuning...')
    train_data_formatted = '\n'.join(
        [format_for_finetuning(
            transaction + f"""\n\n## Content:\n{train_data['dialogue'][i]}\n\n## Topic:\n{train_data['section_header'][i]}\n\n## Summary:""",
            f"""{train_data['section_text'][i]}""",
            system_message
        ) for i in range(len(train_data))]
    )
    validation_data_formatted = '\n'.join(
        [format_for_finetuning(
            transaction + f"""\n\n## Content:\n{validation_data['dialogue'][i]}\n\n## Topic:\n{validation_data['section_header'][i]}\n\n## Summary:""",
            f"""{validation_data['section_text'][i]}""",
            system_message
        ) for i in range(len(validation_data))]
    )

    # Write the formatted data to a file
    print('Writing formatted data to file...')

    with open(path.join(args.formatted_data_dir, f'{args.model_id}_train_data.jsonl'), 'w') as f:
        f.write(train_data_formatted)

    with open(path.join(args.formatted_data_dir, f'{args.model_id}_validation_data.jsonl'), 'w') as f:
        f.write(validation_data_formatted)

    # Set the OpenAI API key and create a client
    openai.api_key = getenv('OPENAI_API_KEY')
    client = OpenAI()

    # Create the training dataset
    train_data_response = client.files.create(
        file=open(path.join(args.formatted_data_dir, f'{args.model_id}_train_data.jsonl'), "rb"),
        purpose="fine-tune"
    )

    validation_data_response = client.files.create(
        file=open(path.join(args.formatted_data_dir, f'{args.finetuned_name}_validation_data.jsonl'), "rb"),
        purpose="fine-tune"
    )

    # Create the fine-tuning job
    job_response = client.fine_tuning.jobs.create(
        training_file=train_data_response.id,
        validation_file=validation_data_response.id,
        model=args.model_id,
        hyperparameters={
            "n_epochs": 1,
        }
    )
    
    # Wait for the fine-tuning job to complete
    job_status = client.fine_tuning.jobs.retrieve(job_response.id)

    while job_status.status != 'succeeded' and job_status.status != 'failed':
        job_status = client.fine_tuning.jobs.retrieve(job_response.id)
        print('Fine-tuning job status: ', job_status.status)
        print(job_status)
        time.sleep(60)

    if job_status.status == 'failed':
        raise Exception('Fine-tuning job failed')

    # Get the name of the fine-tuned model
    finetuned_model = job_status.fine_tuned_model

    # Evaluate the OpenAI model
    print('Evaluating finetuned model')

    # Create the bot from the fine-tuned model
    bot = DialogueBot(model=finetuned_model, system_prompt=system_message)

    # Evaluate the fine-tuned model
    model_outputs, metrics = evaluate_openai_classifications(bot, 
                                              test_data, 
                                              len(test_data), 
                                              transaction=transcation,
                                              args.remove_stop_tokens,
                                              args.intermediate_outputs_dir)    

    # Log the metrics to W&B
        if args.wandb_logging == 'True':
            wandb.log(metrics)

    # Print the metrics to the console
    for key, value in metrics.items():
         print(f'{key}: {value}')

    with open(f"results/{args.model_id}_finetuned_zeroshot_outputs.json", 'w') as f: json.dump(metrics, f)
    np.save(f"results/{args.model_id}_finetuned_zeroshot_outputs.npy", model_outputs)
        
if args.wandb_logging == 'True':
    wandb.finish()        
