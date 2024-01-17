#!/usr/bin/env python3

import os
import evaluate
import numpy as np
import json
import argparse
import torch
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
from typing import Iterable
from tqdm import tqdm
from os import path, makedirs, getenv
from openai_chat_api import DialogueBot

from sklearn.metrics import accuracy_score, f1_score, classification_report

def compute_classification_metrics(predictions: Iterable, 
                            data: Iterable,
                            acc: bool=True,
                            f1: bool=True,
                            wga: bool=True) -> dict:
    """
    Compute accuracy, f1, worst-group accruacy metrics for a set of predictions and references.
    """
    
    labels = []

    for idx in range(len(data['label'])):
        if 'Y' in data['label'][idx]: 
            labels.append(1)
        else:
            labels.append(0)
            

    # labels = labels.cpu()

    
    metric_results = {}

    if acc:
        acc_results = accuracy_score(labels, predictions)

        # Store the results in the metric_results dictionary
        metric_results['acc'] = acc_results

        # print(classification_report(labels, predictions))
    
    else:
        metric_results['acc'] = None

    if f1:
        f1_results = f1_score(labels, predictions)

        # Store the results in the metric_results dictionary
        metric_results['f1'] = f1_results
    
    else:
        metric_results['f1'] = None

    if wga:
        # DEMOGRAPHICS = ["male", "female", "LGBTQ", "christian", "muslim", "other_religions", "black", "white"]
        DEMOGRAPHICS = ["male", "female"]
        group_acc = []
        
        for g in DEMOGRAPHICS:
            g_preds_0 = []
            g_labels_0 = []

            g_preds_1 = []
            g_labels_1 = []
            
            for idx, item in enumerate(data[g]):
                
                if item >= 0.5:
                    if labels[idx] == 1:
                        g_preds_1.append(predictions[idx])
                        g_labels_1.append(labels[idx])
                    else:
                        g_preds_0.append(predictions[idx])
                        g_labels_0.append(labels[idx])

            g_acc_0 = accuracy_score(g_labels_0, g_preds_0)
            g_acc_1 = accuracy_score(g_labels_1, g_preds_1)
            
            group_acc.append(g_acc_0)
            group_acc.append(g_acc_1)
        
        metric_results['wga'] = min(group_acc)
    
    else:
        metric_results['wga'] = None

    return metric_results
def generate_from_prompt(model: AutoModelForCausalLM, 
                         tokenizer: AutoTokenizer, 
                         input_data: str,
                         max_tokens: int=2048,
                         min_new_tokens: int=1,
                         max_new_tokens: int=50) -> str:
    """
    Generate and decode output from a Transformers model using a prompt.
    """

    # Check whether input will not include the end prompt due to context window length, and manually truncate if necessary
    tokenized = tokenizer.encode(input_data)

    # If the input is too long, truncate it to the maximum length minus the length of the end prompt
    #if len(tokenized) > max_tokens:
    #  input = tokenizer.decode(tokenized[:max_tokens-len(tokenizer.encode(end_prompt))-1], skip_special_tokens=True) + end_prompt

    # Calculate the position of the start of the output string
    start_decode = len(tokenizer.encode(input_data, truncation=True, max_length=max_tokens))

    # Encode the input string
    input_ids = tokenizer(input_data, return_tensors='pt', truncation=True, max_length=max_tokens).to(model.device)

    # Generate text from prompt
    with torch.no_grad():
      output = model.generate(**input_ids, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)
    
    # Decode the output string, removing the special tokens and any suffixes
    decoded = tokenizer.decode(output[0][start_decode:])

    return decoded
                             
def evaluate_hf_model(model: AutoModelForCausalLM, 
                      tokenizer: AutoTokenizer, 
                      data: Iterable,
                      examples,
                      shot: str='zero',
                      max_samples: int=None,
                      system_message: str='###',
                      transaction: str='###',
                      max_tokens: int=2048,
                      min_new_tokens: int=1,
                      max_new_tokens: int=50,
                      remove_suffix: str=None,
                      acc: bool=True,
                      f1: bool=True,
                      wga: bool=True) -> dict:
    """
    Evaluate a Hugging Face model on a dataset using three text summarization metrics.
    """
    
    model_outputs = []
    example_1_question = examples['example_1_question']
    example_1_response = examples['example_1_response']
    example_2_question = examples['example_2_question']
    example_2_response = examples['example_2_response']
    example_3_question = examples['example_3_question']
    example_3_response = examples['example_3_response']                          
                          
    # Iterate over the test set
    for idx in tqdm(range(max_samples), desc='Evaluating Hugging Face model'):
  
        # Generate and decode the output string, removing the special tokens and any suffixes
        test_question = f"""\n\n## Comment:\n{data[idx]['text']}\n\n## Response:"""
        
        if 'falcon' not in tokenizer.name_or_path:
            if shot == 'zero':
                chat = [
                    {"role": "user", "content": system_message + transaction + test_question}
                ]
            elif shot == 'one':
                chat = [
                    {"role": "user", "content": system_message + example_1_question},
                    {"role": "assistant", "content": example_1_response},
                    {"role": "user", "content": transaction + test_question},    
                ]
            elif shot == 'two':
                chat = [
                    {"role": "user", "content": system_message + example_1_question},
                    {"role": "assistant", "content": example_1_response},
                    {"role": "user", "content": example_2_question},
                    {"role": "assistant", "content": example_2_response},    
                    {"role": "user", "content": transaction + test_question},    
                ]
            else:
                chat = [
                    {"role": "user", "content": system_message + example_1_question},
                    {"role": "assistant", "content": example_1_response},
                    {"role": "user", "content": example_2_question},
                    {"role": "assistant", "content": example_2_response},
                    {"role": "user", "content": example_3_question},
                    {"role": "assistant", "content": example_3_response},                    
                    {"role": "user", "content": transaction + test_question},    
                ]                
        else:
            if shot == 'zero':
                chat = [
                    {"role": "user", "content": system_message+
                                                transaction+
                                                test_question}
                ]
            elif shot == 'one':
                chat = [
                    {"role": "user", "content": system_message+
                                                example_1_question+
                                                example_1_response+
                                                transaction+
                                                test_question}, 
                ]
            elif shot == 'two':
                chat = [
                    {"role": "user", "content": system_message+
                                                example_1_question+
                                                example_1_response+
                                                example_2_question+
                                                example_2_response+                     
                                                transaction+
                                                test_question},   
                ]
            else:
                chat = [
                    {"role": "user", "content": system_message+
                                                example_1_question+
                                                example_1_response+
                                                example_2_question+
                                                example_2_response+  
                                                example_3_question+
                                                example_3_response+                       
                                                transaction+
                                                test_question},   
                ]
        input_data = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if 'falcon' in tokenizer.name_or_path:
            input_data = input_data.replace('user', 'User:')
            input_data = input_data.replace('assistant', 'Assistant:')
            input_data = input_data.replace('Assistant:!', 'assistant!')
            input_data = input_data.replace('<|im_start|>', '')
            input_data = input_data.replace('<|im_end|>', '')
            
        ## decoding
        decoded = generate_from_prompt(model=model, 
                                       tokenizer=tokenizer, 
                                       input_data=input_data, 
                                       max_tokens=max_tokens,
                                       min_new_tokens=min_new_tokens,
                                       max_new_tokens=max_new_tokens)

        # Remove the suffix if specified - note that Mistral-Instruct models add a </s> suffix to specify the end of the output
        if remove_suffix is not None:
            decoded = decoded.replace(remove_suffix, '')

        # print(decoded)

        lower_decoded = decoded.lower()
        if 'yes' in lower_decoded:
            model_outputs.append(1)
        else:
            model_outputs.append(0)    
    
        # model_outputs.append(int(decoded[0]))
        
    # Compute the ROUGE, BLEU, and BERTscore metrics, comparing the model's responses to the target summaries    
    # metrics = compute_classification_metrics(model_outputs, 
    #                                         data['section_text'][:len(model_outputs)], 
    #                                         rouge=rouge, 
    #                                         bleu=bleu, 
    #                                         bertscore=bertscore)

    
    metrics = compute_classification_metrics(model_outputs, 
                                            data[:len(model_outputs)], 
                                            acc=acc, 
                                            f1=f1, 
                                            wga=wga)
    
    return model_outputs, metrics

def evaluate_openai_model(bot: DialogueBot,
                          data: Iterable, 
                          max_samples: int=None,
                          transaction: str='',
                          remove_stop_tokens: Iterable=None,
                          intermediate_outputs_dir: str=None) -> dict:
    """
    Evaluate an OpenAI model on a dataset using three classification metrics.
    """

    model_outputs = []
    intermediate_idx = 0

    # Load the intermediate outputs - all pickles
    files = [i for i in os.listdir(intermediate_outputs_dir) if i.endswith('.pkl')]

    if files:
        with open(os.path.join(intermediate_outputs_dir, files[0]), 'rb') as f:
            model_outputs.extend(pkl.load(f))

    # If no intermediate outputs, start from the beginning; otherwise, start from the last index
    start_idx = len(model_outputs)

    # Create progress bar
    pbar = tqdm(total=max_samples, desc='Evaluating OpenAI model')
    pbar.update(start_idx)
    
    # Iterate over the test set
    for idx in range(start_idx, max_samples):

        # Create the input string, adding the start and end prompts
        input = transaction+f"""\n\n## Comment:\n{data['text'][idx]}\n\n## Response:"""
      
        # Get the model's response, omitting the system and user prompts
        try:
            output = bot.return_bot_response(input)
        except:
            pkl.dump(model_outputs, open(os.path.join(intermediate_outputs_dir, f'intermediate_outputs.pkl'), 'wb'))
            raise ValueError('OpenAI API error')

        # Remove the stop tokens if specified
        if remove_stop_tokens is not None:
            for token in remove_stop_tokens:
                output = output.replace(token, '')

        stripped_output = output.strip()

        if 'yes' in stripped_output.lower():
            model_outputs.append(1)
        else:
             model_outputs.append(0)

        print(stripped_output)
        
        # model_outputs.append(output.strip())
    
        # Update the progress bar
        pbar.update(1)

    # Compute the classification metrics, comparing the model's responses to the target labels
    metrics = compute_classification_metrics(model_outputs, 
                                            data[:len(model_outputs)])
    return model_outputs, metrics