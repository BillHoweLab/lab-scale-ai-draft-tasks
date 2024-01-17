import json
import argparse
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import Iterable
from datasets import load_dataset

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
    return metric_results
def zeroshot(test_data, client):
    
    #--------------
    # inference
    #--------------
    predictions = []
    gts = []
    for i in tqdm(range(len(test_data))):
    # for i in tqdm(range(100)):
        
        # system = 'Please summarize the following conversation:\n'
        # system = """You are a helpful online content moderator! Please help me detect if an online comment is toxic or non-toxic. I will provide you the comment. """
        system = """You are a helpful online content moderator! Please help me detect if an online comment is toxic or non-toxic. I will provide you the comment. \n\nDoes the following comment contain toxicity? Respond with yes or no only."""
        # test_question = f"""\n\n## Comment:\n{test_data[i]['text']}\n\n## Response:"""
        
        
        dialogue = test_data[i]['text']
        
        # dialogue = test_data[i]['Question'] 
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": dialogue},
            ],
            temperature=0,
            max_tokens=512,
            top_p=1
        )

        # print(response.choices[0].message.content)
        
        lower_output = response.choices[0].message.content.lower()

        if 'yes' in lower_output:
            predictions.append(1)
        else:
            predictions.append(0)
        # gts.append(test_data[i])

    np.save('gpt_zeroshot_outputs.npy', predictions)
    results = compute_classification_metrics(predictions, test_data)    

    print(results)
    
    with open('zeroshot-results.json', 'w') as f:
        json.dump(results, f)
    f.close()

def oneshot(train_data, test_data, client):
    #--------------
    # inference
    #--------------
    predictions = []
    gts = []
    for i in tqdm(range(len(test_data))):
    # for i in tqdm(range(10)):
        # system = 'Summarize the following conversation\n'
        system = """You are a helpful online content moderator! Please help me detect if an online comment is toxic or non-toxic. I will provide you the comment. \n\nDoes the following comment contain toxicity? Respond with yes or no only."""
        # dialogue_example = train_data[0]['dialogue']
        # summary_example = train_data[0]['section_text']
        # dialogue = test_data[i]['dialogue']

        dialogue_example = train_data[11]['text']
        summary_example = train_data[11]['label']
        dialogue = test_data[i]['text']

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": dialogue_example},
                {"role": "assistant", "content": summary_example},
                {"role": "user", "content": dialogue}
            ],
            temperature=0,
            max_tokens=512,
            top_p=1
        )
        # print(response.choices[0].message.content)

        lower_output = response.choices[0].message.content.lower()

        if 'yes' in lower_output:
            predictions.append(1)
        else:
            predictions.append(0)
            
        # gts.append(test_data[i]['section_text'])
        
    results = compute_classification_metrics(predictions, test_data)    

    print(results)
    
    with open('oneshot-results.json', 'w') as f:
        json.dump(results, f)
    f.close()
        
def twoshot(train_data, test_data, client):
    #--------------
    # inference
    #--------------
    predictions = []
    gts = []
    for i in tqdm(range(len(test_data))):
        # system = 'Summarize the following conversation\n'
        system = """You are a helpful online content moderator! Please help me detect if an online comment is toxic or non-toxic. I will provide you the comment. \n\nDoes the following comment contain toxicity? Respond with yes or no only."""
        
        # dialogue_example_1 = train_data[0]['dialogue']
        # summary_example_1 = train_data[0]['section_text']
        # dialogue_example_2 = train_data[1]['dialogue']
        # summary_example_2 = train_data[1]['section_text']
        # dialogue = test_data[i]['dialogue']

        dialogue_example_1 = train_data[11]['text']
        summary_example_1 = train_data[11]['label']
        dialogue_example_2 = train_data[50]['text']
        summary_example_2 = train_data[50]['label']
        dialogue = test_data[i]['text']        

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": dialogue_example_1},
                {"role": "assistant", "content": summary_example_1},
                {"role": "user", "content": dialogue_example_2},
                {"role": "assistant", "content": summary_example_2},
                {"role": "user", "content": dialogue}
            ],
            temperature=0,
            max_tokens=512,
            top_p=1
        )

        lower_output = response.choices[0].message.content.lower()

        if 'yes' in lower_output:
            predictions.append(1)
        else:
            predictions.append(0)

        # predictions.append(response.choices[0].message.content)
        # gts.append(test_data[i]['section_text'])
        
    results = compute_classification_metrics(predictions, test_data)    

    print(results)
    
    with open('fewshot-results.json', 'w') as f:
        json.dump(results, f)
    f.close()


def threeshot(train_data, test_data, client):
    #--------------
    # inference
    #--------------
    predictions = []
    gts = []
    for i in tqdm(range(len(test_data))):
        # system = 'Summarize the following conversation\n'
        system = """You are a helpful online content moderator! Please help me detect if an online comment is toxic or non-toxic. I will provide you the comment. \n\nDoes the following comment contain toxicity? Respond with yes or no only."""
        
        # dialogue_example_1 = train_data[0]['dialogue']
        # summary_example_1 = train_data[0]['section_text']
        # dialogue_example_2 = train_data[1]['dialogue']
        # summary_example_2 = train_data[1]['section_text']
        # dialogue = test_data[i]['dialogue']

        dialogue_example_1 = train_data[11]['text']
        summary_example_1 = train_data[11]['label']
        dialogue_example_2 = train_data[50]['text']
        summary_example_2 = train_data[50]['label']
        dialogue_example_3 = train_data[21]['text']
        summary_example_3 = train_data[21]['label']
        dialogue = test_data[i]['text']        

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": dialogue_example_1},
                {"role": "assistant", "content": summary_example_1},
                {"role": "user", "content": dialogue_example_2},
                {"role": "assistant", "content": summary_example_2},
                {"role": "user", "content": dialogue_example_3},
                {"role": "assistant", "content": summary_example_3},
                {"role": "user", "content": dialogue}
            ],
            temperature=0,
            max_tokens=512,
            top_p=1
        )

        lower_output = response.choices[0].message.content.lower()

        if 'yes' in lower_output:
            predictions.append(1)
        else:
            predictions.append(0)

        # predictions.append(response.choices[0].message.content)
        # gts.append(test_data[i]['section_text'])
        
    results = compute_classification_metrics(predictions, test_data)    

    print(results)
    
    with open('fewshot-results.json', 'w') as f:
        json.dump(results, f)
    f.close()


#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beanham/wikiqa')  
    parser.add_argument('--shottype', type=str, default='True')
    parser.add_argument('--key', type=str, default='openai-key', help='Name of the HuggingFace API token variable name.')
    args = parser.parse_args()
    dataset = args.dataset
    
    #-------------------
    # load data
    #-------------------
    train_data = load_dataset(dataset, split='train')
    validation_data = load_dataset(dataset, split='validation')
    test_data = load_dataset(dataset, split='test')
    client = OpenAI(api_key=args.key)
    
    #--------------
    # inference
    #--------------
    if args.shottype == 'zero':
        print('Zero Shot...')
        zeroshot(test_data, client)
    elif args.shottype == 'one':
        print('One Shot...')
        oneshot(validation_data, test_data, client)
    elif args.shottype == 'two':
        print('Two Shot...')
        twoshot(validation_data, test_data, client)
    else:
        print('Three Shot...')
        threeshot(validation_data, test_data, client)
    
if __name__ == "__main__":
    main()