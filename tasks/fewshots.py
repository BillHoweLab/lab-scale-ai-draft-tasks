import json
import argparse
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import Iterable
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from finetune_functions import get_model_and_tokenizer
from evaluate_functions import evaluate_hf_model
from huggingface_hub import login as hf_login
from os import path, mkdir, getenv
import transformers
import torch

MODEL_SUFFIXES = {
    'openai': '',
    'mistral': '</s>',
    'llama-2': '</s>',
    'falcon': '<|endoftext|>'
}

#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='True')
    parser.add_argument('--dataset', type=str, default='beanham/medsum')
    parser.add_argument('--use_model_prompt_defaults', type=str, default='mistral', help='Whether to use the default prompts for a model')
    parser.add_argument('--nshot', type=str, default='zero', help='The slice of the test dataset to use for fine-tuning.')
    parser.add_argument('--pretrain', type=str, default='True', help='The slice of the test dataset to use for fine-tuning.')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to mount the model on.')
    parser.add_argument('--hf_token_var', type=str, default='test', help='The slice of the test dataset to use for fine-tuning.')
    args = parser.parse_args()
    
    if args.use_model_prompt_defaults:
        args.suffix = MODEL_SUFFIXES[args.use_model_prompt_defaults]
    # if args.hf_token_var:
    #     hf_login(token=getenv(args.hf_token_var))
    
    #-------------------
    # load data
    #-------------------
    print('Getting data...')
    train_data = load_dataset(args.dataset, split='train')
    validation_data = load_dataset(args.dataset, split='validation')
    test_data = load_dataset(args.dataset, split='test')

    #---------------------------
    # prepara prompting examples
    #---------------------------
    index_1 = 11 ## comprehensive example
    index_2 = 50 ## negation    
    index_3 = 21 ## short example
    system_message = """You are a helpful online content moderator! Please help me detect if an online comment is toxic or non-toxic. I will provide you the comment. """
    transaction = """\n\nDoes the following comment contain toxicity? Respond with yes or no only."""
    example_1_question = f"""\n\nExample 1:\n\n## Comment:\n{validation_data[index_1]['text']}\n\n## Response:"""
    example_1_response = f"""{validation_data[index_1]['label']}"""
    example_2_question = f"""\n\nExample 2:\n\n## Comment:\n{validation_data[index_2]['text']}\n\n## Response:"""
    example_2_response = f"""{validation_data[index_2]['label']}"""
    example_3_question = f"""\n\nExample 3:\n\n## Comment:\n{validation_data[index_3]['text']}\n\n## Response:"""
    example_3_response = f"""{validation_data[index_3]['label']}"""
    examples = {
        'example_1_question':example_1_question,
        'example_1_response':example_1_response,
        'example_2_question':example_2_question,
        'example_2_response':example_2_response,
        'example_3_question':example_3_question,
        'example_3_response':example_3_response,        
    }

    print(example_1_question)
    print(example_1_response)

    print(example_2_question)
    print(example_2_response)
    
    print(example_3_question)
    print(example_3_response)
    
    #-------------------
    # load summarizer
    #-------------------
    print('Getting model and tokenizer...')
    model, tokenizer = get_model_and_tokenizer(args.model_id,
                                               gradient_checkpointing=False,
                                               quantization_type='4bit',
                                               device=args.device)
    
    #--------------
    # inference
    #--------------
    #for nshot in ['zero', 'one', 'two', 'three']:
    # for nshot in ['one']:
    #     print(f"{nshot}shot...")
    #     args.nshot = nshot
    model.eval()
    model_outputs, metrics = evaluate_hf_model(model=model,
                                               tokenizer=tokenizer,
                                               data=test_data,
                                               max_samples=len(test_data),
                                               # max_samples=11,
                                               system_message=system_message,
                                               transaction=transaction,
                                               examples = examples,
                                               remove_suffix=args.suffix,
                                               shot = args.nshot)
    print(f'{args.nshot} Results:')
    if args.pretrain == 'True':        
        for k, v in metrics.items():print(f'{k}: {v}')
        with open(f"results/{args.use_model_prompt_defaults}_pretrained_model_{args.nshot}shot_outputs.json", 'w') as f: json.dump(metrics, f)
        np.save(f"results/{args.use_model_prompt_defaults}_pretrained_model_{args.nshot}shot_outputs.npy", model_outputs)
    else:
        for k, v in metrics.items():print(f'{k}: {v}')
        with open(f"results/{args.use_model_prompt_defaults}_finetuned_model_{args.nshot}shot_outputs.json", 'w') as f: json.dump(metrics, f)
        np.save(f"results/{args.use_model_prompt_defaults}_finetuned_model_{args.nshot}shot_outputs.npy", model_outputs)

if __name__ == "__main__":
    main()