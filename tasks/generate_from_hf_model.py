import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
                         system_message: str='###',
                         transaction: str='###',
                         max_tokens: int=974,
                         min_new_tokens: int=25,
                         max_new_tokens: int=50,
                         examples,
                         shot) -> str:
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
