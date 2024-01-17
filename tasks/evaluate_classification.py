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
from generate_from_hf_model import generate_from_prompt

from sklearn.metrics import accuracy_score, f1_score

def compute_classification_metrics(predictions: Iterable, 
                            data: Iterable,
                            acc: bool=True,
                            f1: bool=True,
                            wga: bool=True) -> dict:
    """
    Compute accuracy, f1, worst-group accruacy metrics for a set of predictions and references.
    """

    labels = data['label']

    
    metric_results = {}

    if acc:
        acc_results = accuracy_score(labels, predictions)

        # Store the results in the metric_results dictionary
        metric_results['acc'] = acc_results
    
    else:
        metric_results['acc'] = None

    if f1:
        f1_results = f1_score(labels, predictions)

        # Store the results in the metric_results dictionary
        metric_results['f1'] = f1_results
    
    else:
        metric_results['f1'] = None

    if wga:
        DEMOGRAPHICS = ["male", "female", "LGBTQ", "christian", "muslim", "other_religions", "black", "white"]
        group_acc = []
        
        for g in DEMOGRAPHICS:
            g_preds = []
            g_labels = []
            
            for idx, item in enumerate(data[g]):
                
                if item >= 0.5:
                    g_preds.append(predictions[idx])
                    g_labels.append(labels[idx])

            g_acc = accuracy_score(g_labels, g_preds)
            group_acc.append(g_acc)
        
        metric_results['wga'] = min(group_acc)
    
    else:
        metric_results['wga'] = None

    return metric_results

def evaluate_hf_model(model: AutoModelForCausalLM, 
                      tokenizer: AutoTokenizer, 
                      data: Iterable,
                      input_column: str='article',
                      target_column: str='highlights',
                      max_samples: int=None,
                      start_prompt: str='Summarize the following: ',
                      end_prompt: str='\n Begin summary:',
                      max_tokens: int=974,
                      min_new_tokens: int=25,
                      max_new_tokens: int=50,
                      remove_suffix: str=None,
                      acc: bool=True,
                      f1: bool=True,
                      wga: bool=True) -> dict:
    """
    Evaluate a Hugging Face model on a dataset using three text summarization metrics.
    """
    
    model_outputs = []

    # Iterate over the test set
    for idx in tqdm(range(max_samples), desc='Evaluating Hugging Face model'):

        # Generate and decode the output string, removing the special tokens and any suffixes
        decoded = generate_from_prompt(model, 
                                       tokenizer, 
                                       data[idx][input_column], 
                                       start_prompt, 
                                       end_prompt, 
                                       max_tokens,
                                       min_new_tokens,
                                       max_new_tokens)

        # Remove the suffix if specified - note that Mistral-Instruct models add a </s> suffix to specify the end of the output
        if remove_suffix is not None:
            decoded = decoded.replace(remove_suffix, '')

        model_outputs.append(int(decoded))
        
    # Compute the ROUGE, BLEU, and BERTscore metrics, comparing the model's responses to the target summaries    
    metrics = compute_classification_metrics(model_outputs, 
                                            data[:len(model_outputs)], 
                                            acc=acc, 
                                            f1=f1, 
                                            wga=wga)
    
    return model_outputs, metrics

def evaluate_openai_model(bot: DialogueBot,
                          data: Iterable, 
                          input_column: str,
                          target_column: str,
                          max_samples: int=None,
                          start_prompt: str='Summarize the following: ',
                          end_prompt: str='\n Begin summary:',
                          rouge: bool=True,
                          bleu: bool=True,
                          bertscore: bool=True) -> dict:
    """
    Evaluate an OpenAI model on a dataset using three text summarization metrics.
    """

    model_outputs = []
    
    # Iterate over the test set
    for idx in tqdm(range(max_samples), desc='Evaluating OpenAI model'):

        # Create the input string, adding the start and end prompts
        input = start_prompt + data[idx][input_column] + end_prompt
        
        # Get the model's response, omitting the system and user prompts
        output = bot.return_bot_response(input)
        model_outputs.append(output)
    
    # Compute the ROUGE, BLEU, and BERTscore metrics, comparing the model's responses to the target summaries
    metrics = compute_summarization_metrics(model_outputs, 
                                            data[target_column][:len(model_outputs)], 
                                            rouge=rouge, 
                                            bleu=bleu, 
                                            bertscore=bertscore)
    
    return metrics
