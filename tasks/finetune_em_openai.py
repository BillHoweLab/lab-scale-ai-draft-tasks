#!/usr/bin/env python3

import openai
import torch
import logging
import sys
import datasets
import argparse
import wandb
import json
import time
import numpy as np

from os import path, mkdir, getenv, makedirs
from openai import OpenAI

from finetune_functions import get_dataset_slices
from evaluate_em import evaluate_openai_model_em, transaction, MODEL_SUFFIXES, system_message, examples
from openai_chat_api import DialogueBot

def format_for_finetuning(question: str,
                          assistant_output: str,
                          system_prompt: str='You are a helpful assistant specializing in entity matching.',
                          nshots=0) -> str:
    """
    Format data in JSON for fine-tuning an OpenAI chatbot model.
    """

    chat = [
        {"role": "system", "content": system_prompt}
    ]
    for shot in range(1, nshots + 1):
        if chat[-1]['role'] == 'user':
            chat[-1]['content'] += examples[f'example_{shot}_question']
        else:
            chat.append({"role": "user", "content": examples[f'example_{shot}_question']})
        chat.append({"role": "assistant", "content": examples[f'example_{shot}_response']})
    if chat[-1]['role'] == 'user':
        chat[-1]['content'] += question + transaction
    else:
        chat.append({"role": "user", "content": question + transaction[args.dataset]})
    chat.append({"role": "assistant", "content": assistant_output})

    return json.dumps({"messages": chat})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune an OpenAI model for Entity Matching.')

    # Model ID
    parser.add_argument('--model_id', type=str, default='gpt-3.5-turbo', help='The model ID to fine-tune.')
    parser.add_argument('--finetuned_name', type=str, default='lab-scale-entity-matching', help='The name of the fine-tuned model.')
    parser.add_argument('--system_prompt', type=str, default='You are a helpful assistant specializing in entity matching.', help='The system prompt to use for fine-tuning.')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='isaacOnline/deeds-and-voting-matching', help='The dataset to use for fine-tuning.')
    parser.add_argument('--version', type=str, default='', nargs='?', help='The version of the dataset to use for fine-tuning.')
    parser.add_argument('--input_col', type=str, default='prompt', help='The name of the input column in the dataset.')
    parser.add_argument('--target_col', type=str, default='overall_label', help='The name of the target column in the dataset.')
    parser.add_argument('--train_slice', type=str, default='train', help='The slice of the training dataset to use for fine-tuning.')
    parser.add_argument('--validation_slice', type=str, default='validation[0:25]', help='The slice of the validation dataset to use for fine-tuning.')
    parser.add_argument('--test_slice', type=str, default='test', help='The slice of the test dataset to use for fine-tuning.')
    parser.add_argument('--max_samples', type=int, default=None, help='The maximum number of samples to use for evaluation.')
    parser.add_argument('--remove_stop_tokens', type=str, default=None, help='Stop tokens from the model output.')
    parser.add_argument('--dataset_epochs', type=int, default=1, help='The number of epochs to use for fine-tuning.')

    # Saving arguments
    parser.add_argument('--save_model', type=str, default='True', help='Whether to save the fine-tuned model and tokenizer.')
    parser.add_argument('--results_dir', type=str, help='The directory to save the results to', default='results')
    parser.add_argument('--formatted_data_dir', type=str, help='The directory to save the formatted data to', default='formatted_data')
    parser.add_argument('--intermediate_outputs_dir', type=str, help='The directory to save intermediate outputs to', default='intermediate_outputs')

    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='logs', help='The directory to save the log file.')
    parser.add_argument('--log_level', type=str, default='info', help='The log level to use for fine-tuning.')
    parser.add_argument('--logging_first_step', type=str, default='True', help='Whether to log the first step.')
    parser.add_argument('--logging_steps', type=int, default=1, help='The number of steps between logging.')
    parser.add_argument('--run_name', type=str, default='openai-em-finetune', help='The name of the run, for logging.')

    # W&B logging arguments
    parser.add_argument('--wandb_logging', type=str, default='True', help='Whether to log to W&B.')
    parser.add_argument('--wandb_name', type=str, default='openai-em-finetune', help='The name of the W&B project, for logging.')
    parser.add_argument('--wandb_api_var', type=str, default='WANDB_API_KEY', help='Name of the WandB API key variable name.')

    # Prompt arguments
    parser.add_argument('--start_prompt', type=str, default=None, help='The start prompt to add to the beginning of the input text.')
    parser.add_argument('--suffix', type=str, default=MODEL_SUFFIXES['openai'], help='The suffix to add to the end of the input and target text.')
    parser.add_argument('--max_seq_length', type=int, default=974, help='The maximum sequence length to use for fine-tuning.')
    parser.add_argument('--use_model_prompt_defaults', type=str, default='openai', help='Whether to use the default prompts for a model')

    # Parse arguments
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(536475)

    # Update the start and end prompts if using the model defaults
    if args.start_prompt is None:
        args.suffix = MODEL_SUFFIXES[args.use_model_prompt_defaults]

    # Initialize W&B
    if args.wandb_logging == 'True':
        wandb.login(key=getenv(args.wandb_api_var))
        wandb.init(project=args.wandb_name, 
                   name=args.run_name, 
                   config=args)

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

    # Get dataset splits
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']


    # Format data for fine-tuning
    print('Formatting data for fine-tuning...')
    train_data_formatted = '\n'.join([format_for_finetuning(train_data[args.input_col][i], train_data[args.target_col][i], system_message[args.dataset]) for i in range(len(train_data))])
    validation_data_formatted = '\n'.join([format_for_finetuning(validation_data[args.input_col][i], validation_data[args.target_col][i], system_message[args.dataset]) for i in range(len(validation_data))])

    # Write the formatted data to a file
    print('Writing formatted data to file...')

    with open(path.join(args.formatted_data_dir, f'{args.finetuned_name}_train_data.jsonl'), 'w') as f:
        f.write(train_data_formatted)

    with open(path.join(args.formatted_data_dir, f'{args.finetuned_name}_validation_data.jsonl'), 'w') as f:
        f.write(validation_data_formatted)

    # Set the OpenAI API key and create a client
    openai.api_key = getenv('OPENAI_API_KEY')
    client = OpenAI()

    # Create the training dataset
    data_response = client.files.create(
        file=open(path.join(args.formatted_data_dir, f'{args.finetuned_name}_train_data.jsonl'), "rb"),
        purpose="fine-tune"
    )

    validation_data_response = client.files.create(
        file=open(path.join(args.formatted_data_dir, f'{args.finetuned_name}_validation_data.jsonl'), "rb"),
        purpose="fine-tune"
    )

    # Log finetune start time
    if args.wandb_logging == 'True':
        finetune_start_time = time.time()
        wandb.log({'finetune_time_start': finetune_start_time})

    # Create the fine-tuning job
    job_response = client.fine_tuning.jobs.create(
        training_file=data_response.id,
        validation_file=validation_data_response.id,
        model=args.model_id,
        hyperparameters={
            "n_epochs": args.dataset_epochs,
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

    # Log finetune end time
    if args.wandb_logging == 'True':
        finetune_end_time = time.time()
        wandb.log({'finetune_time_end': finetune_end_time})
        wandb.log({'finetune_time': finetune_end_time - finetune_start_time})

    # Get the name of the fine-tuned model
    finetuned_model = job_status.fine_tuned_model


    # Create intermediate outputs directory
    if not path.exists(args.intermediate_outputs_dir):
        makedirs(args.intermediate_outputs_dir)

    # Log eval time start
    if args.wandb_logging == 'True':
        inference_start_time = time.time()
        wandb.log({'inference_time_start': inference_start_time})

    # Evaluate the OpenAI model
    print('Evaluating OpenAI model on EM task: ', args.model_id)
    chat = [
        {"role": "user", "content": system_message[args.dataset]}
    ]
    bot = DialogueBot(model=finetuned_model, system_prompt=args.system_prompt, history=chat)
    em_metrics, model_output = evaluate_openai_model_em(bot = bot,
                                          data = data['test'],
                                          input_column=args.input_col,
                                          target_column=args.target_col,
                                          max_samples=args.max_samples,
                                          run_name=args.run_name,
                                          remove_stop_tokens=args.remove_stop_tokens,
                                          intermediate_outputs_dir=args.intermediate_outputs_dir,
                                                        dataset=args.dataset)

    # Log eval time end
    if args.wandb_logging == 'True':
        inference_end_time = time.time()
        wandb.log({'inference_time_end': inference_end_time})
        wandb.log({'inference_time': inference_end_time - inference_start_time})

    # Print the metrics to the console
    print('Model EM Metrics:')
    for key, value in em_metrics.items():
        print(f'{key}: {value}')

    # Add the model and dataset names to the metrics dictionary
    metrics = {**vars(args), **em_metrics}

    # Save the metrics to a JSON file
    model_id = args.model_id
    save_path = path.join(args.results_dir, f'{model_id.replace("/", "-")}_em_metrics.json')
    print('Saving EM metrics to: ', save_path)

    # Log the metrics to W&B
    if args.wandb_logging == 'True':
        wandb.log(metrics)
        wandb.log({'Model Output': wandb.Table(dataframe=model_output)})
        wandb.finish()

    if not path.exists(args.results_dir):
        makedirs(args.results_dir)


    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)


    with open(save_path, 'w') as f:
        json.dump(metrics, f, cls=NpEncoder)

    if args.wandb_logging == 'True':
        wandb.finish()
