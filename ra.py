import os
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load the dataset
dataset = load_dataset("wiki40b", 'en', split='train', trust_remote_code=True)

def sample_split_data(data, N=5):
    split_n = []  
    for i, example in enumerate(data):
        split_n.append(example['text'])
        print(f"\n\nSample{i}\n{example['text']}")
        if i == N - 1:
            break
    return split_n

def encode_dataset(dataset_list):
    replacements = {
        '_START_ARTICLE_': 'T1\n',
        '_START_SECTION_': 'T3\n',
        '_START_PARAGRAPH_': 'T2\n',
        '_NEWLINE_': '\nT4\n'
    }
    
    encoded_list = []
    for sample in dataset_list:
        for old, new in replacements.items():
            sample = sample.replace(old, new)
        encoded_list.append(sample)
    
    return encoded_list

def translate(data, target_lang='az_Latn', NLLBversion=0):
    checkpoints = {
        0: 'facebook/nllb-200-1.3B',
        1: 'facebook/nllb-200-distilled-1.3B',
        2: 'facebook/nllb-200-3.3B',
        3: 'facebook/nllb-moe-54b'
    }
    checkpoint = checkpoints.get(NLLBversion, 'facebook/nllb-200-1.3B')
    
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    translation_pipeline = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='en', tgt_lang=target_lang, max_length=400)
    
    translations = [translation_pipeline(text)[0]['translation_text'] for text in data]
    return translations

N = 5  
sampled_data = sample_split_data(dataset, N=N)

encoded_data = encode_dataset(sampled_data)

translated_data = translate(encoded_data, target_lang='az_Latn', NLLBversion=0)

for i, text in enumerate(translated_data):
    print(f"Translated Sample{i}:\n{text}\n")
