#!/usr/bin/env python3

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

# Set the default summarization prompts for each model
DEFAULT_SUMMARIZATION_PROMPTS = {
    'llama-2-base': ('### Summarize: ', '### Begin summary:'),
    'llama-2-chat': ('<s>[INST] <<SYS>> \nYou are a helpful assistant.\n<</SYS>>\n\nSummarize the following: ', ' [/INST]'),
    'gpt2': ('### Summarize: ', '### Begin summary:'),
    'mistral-base': ('### Summarize: ', '### Begin summary:'),
    'mistral-instruct': ('<s>[INST] Summarize the following: ', ' [/INST]'),
    'opt': ('### Summarize: ', '### Begin summary:'),
    'openai': ('Summarize the following: ', '\n Begin summary:'),
    }

def generate_from_prompt(model: AutoModelForCausalLM, 
                         tokenizer: AutoTokenizer, 
                         input_data: str,
                         examples,
                         shot
                         system_message: str='###',
                         transaction: str='###',
                         max_tokens: int=974,
                         min_new_tokens: int=25,
                         max_new_tokens: int=50) -> str:
    """
    Generate and decode output from a Transformers model using a prompt.
    """

    example_1_question = examples['example_1_question']
    example_1_response = examples['example_1_response']
    example_2_question = examples['example_2_question']
    example_2_response = examples['example_2_response']
                             
    # Create the input string, adding the start and end prompts
    if shot == 0:
        chat = [
            {"role": "user", "content": system_message + transaction + input_data}
        ]
        input = tokenizer.apply_chat_template(chat, tokenize=False)
    elif shot == 1:
        chat = [
          {"role": "user", "content": system_message + example_1_question},
          {"role": "assistant", "content": example_1_response},
          {"role": "user", "content": transaction + input_data},    
        ]
        input = tokenizer.apply_chat_template(chat, tokenize=False)
    else:
        chat = [
          {"role": "user", "content": system_message + example_1_question},
          {"role": "assistant", "content": example_1_response},
          {"role": "user", "content": example_2_question},
          {"role": "assistant", "content": example_2_response},    
          {"role": "user", "content": transaction + input_data},    
        ]
        input = tokenizer.apply_chat_template(chat, tokenize=False)

    # Check whether input will not include the end prompt due to context window length, and manually truncate if necessary
    tokenized = tokenizer.encode(input)

    # If the input is too long, truncate it to the maximum length minus the length of the end prompt
    if len(tokenized) > max_tokens:
      input = tokenizer.decode(tokenized[:max_tokens-len(tokenizer.encode(end_prompt))-1], skip_special_tokens=True) + end_prompt

    # Calculate the position of the start of the output string
    start_decode = len(tokenizer.encode(input, truncation=True, max_length=max_tokens))

    # Encode the input string
    input_ids = tokenizer(input, return_tensors='pt', truncation=True, max_length=max_tokens).to(model.device)

    # Generate text from prompt
    with torch.no_grad():
      output = model.generate(**input_ids, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)
    
    # Decode the output string, removing the special tokens and any suffixes
    decoded = tokenizer.decode(output[0][start_decode:])

    return decoded


def compute_summarization_metrics(predictions: Iterable, 
                            references: Iterable,
                            rouge: bool=True,
                            bleu: bool=True,
                            bertscore: bool=True) -> dict:
    """
    Compute ROUGE, BLEU, and BERTscore metrics for a set of predictions and references.
    """

    metric_results = {}

    if rouge:
        rouge = evaluate.load('rouge')

        # Compute ROUGE metrics at the summary level, using the 'rouge1', 'rouge2', and 'rougeL' metrics, aggregating the results
        rouge_results = rouge.compute(predictions=predictions, 
                                    references=references, 
                                    use_aggregator=True)

        # Store the results in the metric_results dictionary
        metric_results['rouge'] = rouge_results
    
    else:
        metric_results['rouge'] = None

    if bleu:
        bleu = evaluate.load('bleu')

        # Compute BLEU metrics at the summary level
        bleu_results = bleu.compute(predictions=predictions, 
                                    references=references)
        
        # Store the results in the metric_results dictionary
        metric_results['bleu'] = bleu_results
    
    else:
        metric_results['bleu'] = None

    if bertscore:
        bertscore = evaluate.load('bertscore')

        # Compute BERTscore metric, using distilbert-base-uncased as the reference model, and averaging the results
        bertscore_results = bertscore.compute(predictions=predictions, 
                                                    references=references, 
                                                    lang='en', 
                                                    model_type="distilbert-base-uncased")
        
        # Store the results in the metric_results dictionary
        metric_results['bertscore'] = {k: np.mean(v) for k, v in bertscore_results.items() if k in ['precision', 'recall', 'f1']}
    
    else:
        metric_results['bertscore'] = None

    return metric_results

def evaluate_hf_model(model: AutoModelForCausalLM, 
                      tokenizer: AutoTokenizer, 
                      data: Iterable,
                      examples,
                      shot,
                      max_samples: int=None,
                      system_message: str='###',
                      transaction: str='###',
                      max_tokens: int=974,
                      min_new_tokens: int=25,
                      max_new_tokens: int=50,
                      remove_suffix: str=None,
                      rouge: bool=True,
                      bleu: bool=True,
                      bertscore: bool=True) -> dict:
    """
    Evaluate a Hugging Face model on a dataset using three text summarization metrics.
    """
    
    model_outputs = []

    # Iterate over the test set
    for idx in tqdm(range(max_samples), desc='Evaluating Hugging Face model'):
  
        # Generate and decode the output string, removing the special tokens and any suffixes
        input_data = f"""\n\n## Dialogue:\n{data[idx]['dialogue']}\n\n## Topic:\n{tdata[idx]['section_header']}\n\n## Summary:"""
        decoded = generate_from_prompt(model=model, 
                                       tokenizer=tokenizer, 
                                       input_data=input_data, 
                                       system_message=system_message, 
                                       transaction=transaction, 
                                       examples=examples,
                                       shot=shot,
                                       max_tokens=max_tokens,
                                       min_new_tokens=min_new_tokens,
                                       max_new_tokens=max_new_tokens)

        # Remove the suffix if specified - note that Mistral-Instruct models add a </s> suffix to specify the end of the output
        if remove_suffix is not None:
            decoded = decoded.replace(remove_suffix, '')

        model_outputs.append(decoded)
        
    # Compute the ROUGE, BLEU, and BERTscore metrics, comparing the model's responses to the target summaries    
    metrics = compute_summarization_metrics(model_outputs, 
                                            data['section_text'][:len(model_outputs)], 
                                            rouge=rouge, 
                                            bleu=bleu, 
                                            bertscore=bertscore)
    
    return model_outputs, metrics
