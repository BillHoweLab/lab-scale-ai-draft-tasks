import numpy as np
import argparse
import json
import warnings
from os import path, makedirs, getenv
from typing import Iterable
from typing import List
import pickle as pkl
import os


import torch
import wandb
import numpy as np
import pandas as pd
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from openai_chat_api import DialogueBot
from generate_from_hf_model import generate_from_prompt
from finetune import QUANZATION_MAP


EM_INSTRUCTION = {
    'isaacOnline/deeds-and-voting-matching':"""Your task is to compare pairs of names and addresses for people from Mecklenburg County, North Carolina,
  to determine if they match. The names and address are derived from public records. Review each pair to see if they
  refer to the same person, considering variations in name presentation and address details. Uncommon names or similar
  addresses might indicate a match, but be cautious, as some names are more common than others.
Do these records refer to the same person? Output only one token.

  Enter 'y' if they do
  Enter 'n' if they do not

""",
'isaacOnline/rel-text':"""Your task is to compare records from two datasets, to determine if they refer to the same 
entity. The records may have different columns and formats, or may all been in the same format. Columns may also 
be spelled differently in one dataset than in the other. Any missing values are represented with the text "NA".
Do these records refer to the same entity? Output only one token.

  Enter 'y' if they do
  Enter 'n' if they do not

"""
}

EM_ONE_SHOT = {
    'isaacOnline/deeds-and-voting-matching':"""\nHere is an example of how to perform the task. 
Name 1: DOAN TIFFANY A. JORDAN
Name 2: DOAN ETHAN TRUONG
Address 1: 4652 CEDAR ROCK DR CHARLOTTE NC
Address 2: 4652 CEDAR ROCK DR CHARLOTTE NC

What is the correct label? n
""",
    'isaacOnline/rel-text': """\nHere is an example of how to perform the task.
LEFT text: This paper describes an efficient optimistic concurrency control scheme for use in distributed database systems in which objects are cached and manipulated at client machines while persistent storage and transactional support are provided by servers. The scheme provides both serializability and external consistency for committed transactions; it uses loosely synchronized clocks to achieve global serialization. It stores only a single version of each object, and avoids maintaining any concurrency control information on a per-object basis; instead, it tracks recent invalidations on a per-client basis, an approach that has low in-memory space overhead and no per-object disk overhead.
RIGHT text: NA

LEFT venue: NA
RIGHT venue: international conference on management of data

LEFT id: NA
RIGHT id: 1595

LEFT title: NA
RIGHT title: efficient optimistic concurrency control using loosely synchronized clocks

LEFT year: NA
RIGHT year: 1995

LEFT authors: NA
RIGHT authors: atul adya , robert gruber , barbara liskov , umesh maheshwari

What is the correct label? y
    """
}
EM_TWO_SHOT = {
    'isaacOnline/deeds-and-voting-matching':"""Here is another example of how to perform the task. 
Name 1: GILLARD JC
Name 2: GILLARD JR J C
Address 1: 3802 RICELAND PL UNINC NC
Address 2: 3802 RICELAND PL CHARLOTTE NC

What is the correct label? y
""",
    'isaacOnline/rel-text': """Here is another example of how to perform the task.
LEFT text: This paper takes the next logical step: It considers the use of timestamping for capturing transaction and valid time in the context of transactions. The paper initially identifies and analyzes several problems with straightforward timestamping, then proceeds to propose a variety of techniques aimed at solving these problems. Timestamping the results of a transaction with the commit time of the transaction is a promising approach. The paper studies how this timestamping may be done using a spectrum of techniques. While many database facts are valid until now, the current time, this value is absent from the existing temporal types. Techniques that address this problem using different substitute values are presented. Using a stratum architecture, the performance of the different proposed techniques are studied. Although querying and modifying time-varying data is accompanied by a number of subtle problems, we present a comprehensive approach that provides application programmers with simple, consistent, and efficient support for modifying bitemporal databases in the context of user transactions.

RIGHT text: NA

LEFT venue: NA
RIGHT venue: the vldb journal -- the international journal on very large data bases

LEFT id: NA
RIGHT id: 1213

LEFT title: NA
RIGHT title: priority assignment in real-time active databases

LEFT year: NA
RIGHT year: 1996

LEFT authors: NA
RIGHT authors: rajendran m. sivasankaran , john a. stankovic , don towsley , bhaskar purimetla , krithi ramamritham

What is the correct label? n
"""
} 

EM_THREE_SHOT = {
    'isaacOnline/deeds-and-voting-matching':"""Here is another example of how to perform the task. 
Name 1: WHITE DEBRA
Name 2: WHITE DEBRA
Address 1: 4634 DEER CROSS TL CHARLOTTE NC
Address 2: 3130 CRISP WOOD LN CHARLOTTE NC

What is the correct label? n
""",
    'isaacOnline/rel-text': """Here is another example of how to perform the task.
LEFT text: Recent studies have shown that cache-conscious indexes outperform conventional main memory indexes. Cache-conscious indexes focus on better utilization of each cache line for improving search performance of a single lookup. None has exploited cache spatial and temporal locality between consecutive lookups. We show that conventional indexes, even ""cache-conscious"" ones, suffer from significant cache thrashing between accesses. Such thrashing can impact the performance of applications such as stream processing and query operations such as index-nested-loops join.

RIGHT text: NA

LEFT venue: NA
RIGHT venue: very large data bases

LEFT id: NA
RIGHT id: 1408

LEFT title: NA
RIGHT title: efficient index structures for string databases

LEFT year: NA
RIGHT year: 2001

LEFT authors: NA
RIGHT authors: tamer kahveci , ambuj k. singh

What is the correct label? n
"""
}
EM_TRANSITION = 'Here is the example that needs to be classified. Please respond with only one token after being asked for the correct label. '
# Default chat tokens, end prompts, and suffixes for each model
EM_MODEL_CHAT_TOKENS = {
    'openai': '',
    'mistral': '<s>[INST] ',
    'llama-2': '<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n',
    'falcon': 'A helpful assistant.\nUser: ',
    'opt-finetune': '',
}

EM_MODEL_END_PROMPTS = {
    'openai': ' What is the correct label?',
    'mistral': ' What is the correct label? [/INST]',
    'llama-2': ' What is the correct label? [/INST]',
    'falcon': ' What is the correct label?\nAssistant:',
    'opt-finetune': ' What is the correct label?',
}

EM_MODEL_SUFFIXES = {
    'openai': '',
    'mistral': '</s>',
    'llama-2': '</s>',
    'falcon': '<|endoftext|>',
    'opt-finetune': '</s>',
}

def normalize_answer(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def clean_response(text, positive_value, negative_value):
    # Tokenize
    gold_toks = get_tokens(text)

    # Filter down to only valid response tokens
    gold_toks = [tok for tok in gold_toks if tok in [positive_value, negative_value]]

    # Filter out cases where there are multiple response tokens (or no response tokens) in output
    response = gold_toks[0] if len(gold_toks) == 1 else ''

    if response == positive_value:
        return 1
    elif response == negative_value:
        return 0
    else:
        return -1

def calculate_em_metrics(preds: List[int], truths: List[int], formattings: List[str]):
    """
    Calculate
    """

    # Compute the final metrics
    truths = np.array(truths)
    preds = np.array(preds)
    formattings = np.array(formattings)

    # Compute the metrics
    N = truths.shape[0]
    accuracy = ((truths == preds)).sum() / N
    TP = ((preds == 1) & (truths == 1)).sum()
    FP = ((preds == 1) & (truths == 0)).sum()
    FN = ((preds == 0) & (truths == 1)).sum()
    TN = ((preds == 0) & (truths == 0)).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    # Formatting
    unresolvable_total = (formattings == 'UNRESOLVABLE').sum()
    unresolvable_positives = ((formattings == 'UNRESOLVABLE') & (truths == 1)).sum()
    unresolvable_negatives = ((formattings == 'UNRESOLVABLE') & (truths == 0)).sum()
    unresolvable_rate = unresolvable_total / N

    resolvable_total = (formattings == 'RESOLVABLE').sum()
    resolvable_positives = ((formattings == 'RESOLVABLE') & (truths == 1)).sum()
    resolvable_negatives = ((formattings == 'RESOLVABLE') & (truths == 0)).sum()
    resolvable_rate = resolvable_total / N

    correct_fmt_total = (formattings == 'CORRECT').sum()
    correct_fmt_positives = ((formattings == 'CORRECT') & (truths == 1)).sum()
    correct_fmt_negatives = ((formattings == 'CORRECT') & (truths == 0)).sum()
    correct_fmt_rate = correct_fmt_total / N


    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,

        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,

        'unresolvable_total':unresolvable_total,
        'unresolvable_positives':unresolvable_positives,
        'unresolvable_negatives':unresolvable_negatives,
        'unresolvable_rate':unresolvable_rate,

        'resolvable_total' :resolvable_total,
        'resolvable_positives':resolvable_positives,
        'resolvable_negatives':resolvable_negatives,
        'resolvable_rate':resolvable_rate,

        'correct_fmt_total':correct_fmt_total,
        'correct_fmt_positives':correct_fmt_positives,
        'correct_fmt_negatives':correct_fmt_negatives,
        'correct_fmt_rate':correct_fmt_rate,
    }


############

def evaluate_hf_model_em(model: AutoModelForCausalLM,
                         tokenizer: AutoTokenizer,
                         data: Iterable,
                         input_column: str = 'input',
                         target_column: str = 'output',
                         max_samples: int = None,
                         max_tokens: int = 974,
                         min_new_tokens: int = 1,
                         max_new_tokens: int = 100,
                         remove_suffix: str = None,
                         save_output_dir: str=None,
                         run_name: str='',
                         start_prompt: str = '### Consider the following question with context: ',
                         end_prompt: str = ' ### Please answer with one of the options listed in the brackets:',
                         positive_value: str = 'y',
                         negative_value: str = 'n',
                         remove_stop_tokens: Iterable=None,
                         results_dir: str = '',
                         ) -> dict:
    """
    Evaluate a Hugging Face model on a Entity Matching task.
    """
    preds = []
    truths = []
    full_responses = []
    formattings = []

    if not max_samples:
        max_samples = len(data)

    # Iterate over the test set
    for idx in tqdm(range(min(max_samples, len(data))), desc='Evaluating EM model'):
        question = data[idx][input_column]
        ground_truth = data[idx][target_column]

        # Generate and decode the output string, removing the special tokens and any suffixes
        decoded = generate_from_prompt(model=model,
                                       tokenizer=tokenizer,
                                       input_data=question,
                                       start_prompt=start_prompt,
                                       end_prompt=end_prompt,
                                       max_tokens=max_tokens,
                                       min_new_tokens=min_new_tokens,
                                       max_new_tokens=max_new_tokens)
        full_responses.append(decoded)

        # Remove the suffix if specified
        if remove_suffix is not None and remove_suffix in decoded:
            decoded = decoded.split(remove_suffix)[0]

        # Remove the stop tokens if specified
        if remove_stop_tokens is not None:
            for token in remove_stop_tokens:
                decoded = decoded.replace(token, '')

        # Classify whether formatting is correct (only one output token)
        formatting = 'CORRECT' if decoded in [positive_value, negative_value] else 'UNRESOLVABLE'

        # Since responses may include more tokens than just the requested response token (e.g. 'y' or 'n'),
        # extract the first token from the response
        decoded = clean_response(decoded, positive_value, negative_value)
        ground_truth = clean_response(ground_truth, positive_value, negative_value)


        # Classify whether formatting is resolvable (has a correct token that can be pulled out)
        formatting = 'RESOLVABLE' if decoded in [0, 1] and formatting != 'CORRECT' else formatting

        # Add the decoded and ground truth responses to the list
        preds.append(decoded)
        truths.append(ground_truth)
        formattings.append(formatting)

    metrics = calculate_em_metrics(preds, truths, formattings)

    # Save model outputs to dataframe
    output_df = pd.DataFrame(np.column_stack([data[input_column][:len(preds)],
                                              data[target_column][:len(preds)],
                                              preds,
                                              full_responses,
                                              formattings]),
                             columns=['input', 'label', 'output', 'full_response', 'formatting'])
    if save_output_dir:
        output_df.to_csv(path.join(save_output_dir, f'{run_name}_outputs.csv'))

    return metrics, output_df


def evaluate_openai_model_em(bot: DialogueBot,
                             data: Iterable,
                             input_column: str = 'input',
                             target_column: str = 'output',
                             max_samples: int = None,
                             start_prompt: str = '### Consider the following question with context: ',
                             end_prompt: str = ' ### Please answer with one of the options listed in the brackets:',
                             positive_value: str = 'y',
                             negative_value: str = 'n',
                             save_output_dir=None,
                             results_dir=None,
                             run_name='',
                             remove_stop_tokens=False,
                             intermediate_outputs_dir=None
                             ) -> dict:
    """
    Evaluate an OpenAI model on a dataset using EM metrics.


    """
    preds = []
    truths = []
    full_responses = []
    formattings = []

    # Load the intermediate outputs - all pickles
    files = [i for i in os.listdir(intermediate_outputs_dir) if i.endswith('.pkl')]

    if files:
        with open(os.path.join(intermediate_outputs_dir, 'preds.pkl'), 'rb') as f:
            preds.extend(pkl.load(f))
        with open(os.path.join(intermediate_outputs_dir, 'full_responses.pkl'), 'rb') as f:
            preds.extend(pkl.load(f))
        with open(os.path.join(intermediate_outputs_dir, 'formattings.pkl'), 'rb') as f:
            preds.extend(pkl.load(f))
        min_size = min(len(preds), len(full_responses), len(formattings))
        if min_size != len(preds) or min_size != len(full_responses) or min_size != len(formattings):
            warnings.warn('Intermediate outputs are not the same size. Truncating to the smallest size.')
            preds = preds[:min_size]
            full_responses = full_responses[:min_size]
            formattings = formattings[:min_size]

        truths = [clean_response(data[i][target_column], positive_value, negative_value) for i in range(min_size)]

    # If no intermediate outputs, start from the beginning; otherwise, start from the last index
    start_idx = len(truths)

    if not max_samples:
        max_samples = len(data)

    # Iterate over the dataset
    for idx in tqdm(range(min(max_samples, len(data))), desc='Evaluating OpenAI EM model'):
        # Create the input string
        input = start_prompt + data[idx][input_column] + end_prompt
        ground_truth = data[idx][target_column]


        # Get the model's response, omitting the system and user prompts
        try:
            decoded = bot.return_bot_response(input)
        except Exception as e:
            pkl.dump(preds, open(os.path.join(intermediate_outputs_dir, f'preds.pkl'), 'wb'))
            pkl.dump(full_responses, open(os.path.join(intermediate_outputs_dir, f'full_responses.pkl'), 'wb'))
            pkl.dump(formattings, open(os.path.join(intermediate_outputs_dir, f'formattings.pkl'), 'wb'))
            raise ValueError(f'OpenAI API error: {e.message}')
        full_responses.append(decoded)

        # Remove the stop tokens if specified
        if remove_stop_tokens is not None:
            for token in remove_stop_tokens:
                decoded = decoded.replace(token, '')

        # Classify whether formatting is correct (only one output token)
        formatting = 'CORRECT' if decoded in [positive_value, negative_value] else 'UNRESOLVABLE'

        # Since responses may include more tokens than just the requested response token (e.g. 'y' or 'n'),
        # extract the first token from the response
        decoded = clean_response(decoded, positive_value, negative_value)
        ground_truth = clean_response(ground_truth, positive_value, negative_value)

        # Classify whether formatting is resolvable (has a correct token that can be pulled out)
        formatting = 'RESOLVABLE' if decoded in [0, 1] and formatting != 'CORRECT' else formatting

        # Add the decoded and ground truth responses to the list
        preds.append(decoded)
        truths.append(ground_truth)
        formattings.append(formatting)

    metrics = calculate_em_metrics(preds, truths, formattings)

    # Save model outputs to dataframe
    output_df = pd.DataFrame(np.column_stack([data[input_column][:len(preds)],
                                              data[target_column][:len(preds)],
                                              preds,
                                              full_responses,
                                              formattings]),
                             columns=['input', 'label', 'output', 'full_response', 'formatting'])
    if save_output_dir:
        output_df.to_csv(path.join(save_output_dir, f'{run_name}_outputs.csv'))

    return metrics, output_df


if __name__ == '__main__':
    warnings.warn("YOU ARE RUNNING A SCRIPT THAT HAS NOT BEEN ADAPTED TO EM ")

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a model on an EM task.')

    # Model arguments
    parser.add_argument('--model_type', type=str, help='The type of model to evaluate (Huggingface or OpenAI)',
                        default='hf')
    parser.add_argument('--hf_model_id', type=str, help='The Huggingface model to evaluate',
                        default='mistralai/Mistral-7B-Instruct-v0.1')
    parser.add_argument('--oai_model_id', type=str, help='The OpenAI model ID to use', default='gpt-3.5-turbo')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, help='The dataset to evaluate on', default='isaacOnline/deeds-and-voting-matching')
    parser.add_argument('--dataset_revision', type=str, help='The revision of the dataset to use', default=None)
    parser.add_argument('--split', type=str, help='The split of the dataset to evaluate on', default='test')
    parser.add_argument('--input_column', type=str, help='The name of the input column in the dataset',
                        default='prompt')
    parser.add_argument('--target_column', type=str, help='The name of the target column in the dataset',
                        default='overall_label')
    parser.add_argument('--max_samples', type=int, help='The maximum number of samples to evaluate', default=None)

    # Prompt arguments
    parser.add_argument('--use_model_prompt_defaults', type=str, help='Whether to use the default prompts for a model',
                        default='mistral')
    parser.add_argument('--system_prompt', type=str, help='The system prompt for the model',
                        default='You are a helpful assistant.')
    parser.add_argument('--start_prompt', type=str, help='The start prompt for the model', default=None)
    parser.add_argument('--end_prompt', type=str, help='The end prompt for the model', default=' Label?')
    parser.add_argument('--max_tokens', type=int, help='The maximum number of tokens to generate', default=974)
    parser.add_argument('--remove_suffix', type=str, help='The suffix to remove from the generated output', default=None)
    parser.add_argument('--remove_stop_tokens', type=str, help='Stop tokens to remove from generated output separated by +', default='.')

    # Few-shot arguments
    parser.add_argument('--shots', type=int, help='The number of shots to use for the model', default=0)
    parser.add_argument('--first_shot', type=str, help='The first shot to use for the model', default=None)
    parser.add_argument('--second_shot', type=str, help='The second shot to use for the model', default=None)
    parser.add_argument('--third_shot', type=str, help='The third shot to use for the model', default=None)
    parser.add_argument('--transition', type=str, help='The transition to use between shots', default=EM_TRANSITION)

    # PEFT arguments
    parser.add_argument('--peft_model', type=bool, help='Whether to use a PEFT model', default=False)
    parser.add_argument('--peft_dir', type=str, help='The path to the PEFT model config file', default='')
    parser.add_argument('--four_bit', type=bool, help='Whether to use a 4-bit PEFT model', default=True)
    parser.add_argument('--eight_bit', type=bool, help='Whether to use an 8-bit PEFT model', default=False)

    # Generation arguments
    parser.add_argument('--min_new_tokens', type=int, help='The minimum number of new tokens to generate', default=1)
    parser.add_argument('--max_new_tokens', type=int, help='The maximum number of new tokens to generate', default=10)

    # Environment and reproducibility arguments
    parser.add_argument('--device', type=str, help='The device to use for inference', default='cuda:0')
    parser.add_argument('--seed', type=int, help='The random seed to use', default=42)
    parser.add_argument('--results_dir', type=str, help='The directory to save the results to', default='results')
    parser.add_argument('--intermediate_outputs_dir', type=str, help='The directory to save the intermediate outputs to', default='intermediate_outputs')
    parser.add_argument('--run_name', type=str, default='fact_checking_eval', help='The name of the project, for logging.')

    # W&B logging arguments
    parser.add_argument('--wandb_logging', type=str, default='True', help='Whether to log to W&B.')
    parser.add_argument('--wandb_name', type=str, default='em_eval', help='The name of the W&B project, for logging.')
    parser.add_argument('--wandb_api_var', type=str, default='WANDB_API_KEY',
                        help='Name of the WandB API key variable name.')

    # OpenAI Key
    parser.add_argument('--openai_api_var', type=str, default='OPENAI_API_KEY',
                        help='Name of the WandB API key variable name.')

    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)

    # Update the run name
    args.run_name = f'{args.run_name}_{args.shots}-shot'

    # Initialize W&B
    if args.wandb_logging == 'True':
        wandb.login(key=getenv(args.wandb_api_var))
        wandb.init(project=args.wandb_name,
                   name=args.run_name,
                   config=args)

    # Set openai Api key var
    if args.openai_api_var != 'OPENAI_API_KEY':
        os.environ['OPENAI_API_KEY'] = getenv(args.openai_api_var)

    # Create results directory
    if not path.exists(args.results_dir):
        makedirs(args.results_dir)

    # Update the start and end prompts if using the model defaults
    if args.start_prompt is None:
        args.start_prompt = EM_INSTRUCTION[args.dataset]
    if args.use_model_prompt_defaults:
        args.start_prompt = EM_MODEL_CHAT_TOKENS[args.use_model_prompt_defaults] + args.start_prompt
        args.end_prompt = EM_MODEL_END_PROMPTS[args.use_model_prompt_defaults]
        args.remove_suffix = EM_MODEL_SUFFIXES[args.use_model_prompt_defaults]


    # Add shots to the start prompt if specified
    if args.shots > 0:

        if args.shots == 1:
            if args.first_shot is None:
                args.first_shot = EM_ONE_SHOT[args.dataset]
            args.start_prompt = args.start_prompt + args.first_shot + args.transition
        elif args.shots == 2:
            if args.second_shot is None:
                args.second_shot = EM_TWO_SHOT[args.dataset]
            args.start_prompt = args.start_prompt + args.first_shot + args.second_shot + args.transition
        elif args.shots == 3:
            if args.third_shot is None:
                args.third_shot = EM_THREE_SHOT[args.dataset]
            args.start_prompt = args.start_prompt + args.first_shot + args.second_shot + args.third_shot + args.transition
        else:
            raise ValueError('Invalid number of shots: ', args.shots)

    # Create list of stop tokens to remove
    if args.remove_stop_tokens:
        args.remove_stop_tokens = args.remove_stop_tokens.split('+')

    # Load the test split of the dataset
    print('Loading dataset: ', args.dataset)
    if args.dataset_revision:
        data = load_dataset(args.dataset, args.dataset_revision, split=args.split)
    else:
        data = load_dataset(args.dataset, split=args.split)

    # HF model
    if args.model_type == 'hf':
        # Load the Hugging Face model and tokenizer
        print('Loading Hugging Face model: ', args.hf_model_id)
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load the quantized model in the specified precision
        if args.four_bit:
            model = AutoModelForCausalLM.from_pretrained(args.hf_model_id, quantization_config=QUANZATION_MAP['4bit'])

        elif args.eight_bit:
            model = AutoModelForCausalLM.from_pretrained(args.hf_model_id, quantization_config=QUANZATION_MAP['8bit'])

        # If the model is not a quantized model, load the Hugging Face model and tokenizer
        else:
            model = AutoModelForCausalLM.from_pretrained(args.hf_model_id).to(args.device)

        # If the model is a PEFT model, load the PEFT model and tokenizer
        if args.peft_model:
            # Get the PEFT model
            model = PeftModel.from_pretrained(model, args.peft_dir)

        # Set the model to evaluation mode
        model.eval()

        # Evaluate the Hugging Face model
        print('Evaluating Hugging Face model on EM task: ', args.hf_model_id)
        em_metrics, model_output = evaluate_hf_model_em(model=model,
                                          tokenizer=tokenizer,
                                          data=data,
                                          input_column=args.input_column,
                                          target_column=args.target_column,
                                          max_samples=args.max_samples,
                                          start_prompt=args.start_prompt,
                                          end_prompt=args.end_prompt,
                                          max_tokens=args.max_tokens,
                                          min_new_tokens=args.min_new_tokens,
                                          max_new_tokens=args.max_new_tokens,
                                          remove_suffix=args.remove_suffix,
                                          results_dir=args.results_dir,
                                          run_name=args.run_name,
                                          remove_stop_tokens=args.remove_stop_tokens)

    # OpenAI model
    elif args.model_type == 'openai':
        # NOTE: OpenAI Diaglogue bot QandA task has not been tested
        # TODO: Test

        args.intermediate_outputs_dir = f'{args.intermediate_outputs_dir}_{args.shots}-shot'

        # Create intermediate outputs directory
        if not path.exists(args.intermediate_outputs_dir):
            makedirs(args.intermediate_outputs_dir)

        # Evaluate the OpenAI model
        print('Evaluating OpenAI model on EM task: ', args.oai_model_id)
        bot = DialogueBot(model=args.oai_model_id, system_prompt=args.system_prompt)
        em_metrics, model_output = evaluate_openai_model_em(bot = bot,
                                              data = data,
                                              input_column=args.input_column,
                                              target_column=args.target_column,
                                              max_samples=args.max_samples,
                                              start_prompt=args.start_prompt,
                                              end_prompt=args.end_prompt,
                                              results_dir=args.results_dir,
                                              run_name=args.run_name,
                                              remove_stop_tokens=args.remove_stop_tokens,
                                              intermediate_outputs_dir=args.intermediate_outputs_dir)

    else:
        raise ValueError('Invalid model type: ', args.model_type)

    # Print the metrics to the console
    print('Model EM Metrics:')
    for key, value in em_metrics.items():
        print(f'{key}: {value}')

    # Add the model and dataset names to the metrics dictionary
    metrics = {**vars(args), **em_metrics}

    # Save the metrics to a JSON file
    model_id = args.hf_model_id if args.model_type == 'hf' else args.oai_model_id
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
