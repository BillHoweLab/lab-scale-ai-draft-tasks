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
from finetune import get_model_and_tokenizer
from evaluate_summarization import evaluate_hf_model
import transformers
import torch

MODEL_CHAT_TOKENS = {
    'openai': '',
    'mistral': '<s>[INST] ',
    'llama-2': '<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n',
    'falcon': 'A helpful assistant.\nUser: ',
    'opt-finetune': '',
}

MODEL_END_PROMPTS = {
    'openai': ' ',
    'mistral': '[/INST] ',
    'llama-2': '[/INST] ',
    'falcon': '\nAssistant: ',
    'opt-finetune': ' ',
}

MODEL_SUFFIXES = {
    'openai': '',
    'mistral': '</s>',
    'llama-2': '</s>',
    'falcon': '<|endoftext|>',
    'opt-finetune': '</s>',
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
    parser.add_argument('--dataset', type=str, default='beanham/wikiqa')
    parser.add_argument('--input_column', type=str, default='Question')
    parser.add_argument('--target_column', type=str, default='Sentence')
    parser.add_argument('--start_prompt', type=str, default='Please summarize the following conversation:\n\n')
    parser.add_argument('--end_prompt', type=str, default='\n\nBegin summary: ')
    parser.add_argument('--suffix', type=str, default='</s>', help='The suffix to add to the end of the input and target text.')
    parser.add_argument('--use_model_prompt_defaults', type=str, default='mistral', help='Whether to use the default prompts for a model')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to mount the model on.')
    parser.add_argument('--shot', type=str, default='0', help='The device to mount the model on.')
    args = parser.parse_args()
    
    #-------------------
    # load data
    #-------------------
    print('Getting data...')
    train_data = load_dataset(args.dataset, split='train')
    validation_data = load_dataset(args.dataset, split='validation')
    test_data = load_dataset(args.dataset, split='test')
    oneshot = f"\n\n## Here is an example:\n{train_data[0]['dialogue']}.\n\nSummary:\n{train_data[0]['section_text']}"
    twoshot = f"\n\n## Here is another example:\n{train_data[1]['dialogue']}.\n\nSummary:\n{train_data[1]['section_text']}"
    threeshot = f"\n\n## Here is another example:\n{train_data[2]['dialogue']}.\n\nSummary:\n{train_data[2]['section_text']}"
    transition = "\n\nNow please summarize the following conversation:\n\n"
    
    if args.use_model_prompt_defaults:

        args.start_prompt = MODEL_CHAT_TOKENS[args.use_model_prompt_defaults] + args.start_prompt
        args.end_prompt = args.end_prompt + MODEL_END_PROMPTS[args.use_model_prompt_defaults]
        args.suffix = MODEL_SUFFIXES[args.use_model_prompt_defaults]    

    # oneshot
    if int(args.shot) == 1:        
        args.start_prompt = args.start_prompt+oneshot+transition
    if int(args.shot) == 2:        
        args.start_prompt = args.start_prompt+oneshot+twoshot+transition
    if int(args.shot) == 3:
        args.start_prompt = args.start_prompt+oneshot+twoshot+threeshot+transition
        
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
    model.eval()
    
    print('Evaluating model on ROUGE, BLEU, and BERTScore...')
    model_outputs, metrics = evaluate_hf_model(model,
                                               tokenizer,
                                               test_data,
                                               input_column=args.input_column,
                                               target_column=args.target_column,
                                               max_samples=len(test_data),
                                               start_prompt=args.start_prompt,
                                               end_prompt=args.end_prompt,
                                               remove_suffix=args.suffix)
    
    for k, v in metrics.items():
        print(f'{k}: {v}')
        
    np.save(f"{args.model_id.split('/')[1]}_pretrained_model_outputs.npy", model_outputs)
    
if __name__ == "__main__":
    main()
