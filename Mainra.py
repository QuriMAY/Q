import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

dataset = load_dataset("wiki40b", 'en', trust_remote_code=True)
data=dataset['train']

def process_text(text):
    special_words = ['_START_ARTICLE_', '_START_SECTION_', '_START_PARAGRAPH_', '_NEWLINE_']
    ordered_special_words = []
    split_text = []
    
    for word in special_words:
        if word in text:
            ordered_special_words.append(word)  
            text = text.replace(word, '\n\n')  
    
    split_text = text.split('\n\n')
    
    split_text = [item for item in split_text if item]
    ordered_special_words = [item for item in ordered_special_words if item]
    
    return ordered_special_words, split_text

def reconstruct_texts(ordered_special_words_list, split_text_list):
    reconstructed_texts = []
    for ordered_special_words, split_text in zip(ordered_special_words_list, split_text_list):
        reconstructed_text = ""
        for word, text_segment in zip(ordered_special_words, split_text):
            reconstructed_text += word + text_segment
        if len(split_text) > len(ordered_special_words):
            reconstructed_text += "".join(split_text[len(ordered_special_words):])
        
        reconstructed_texts.append(reconstructed_text)
    
    return reconstructed_texts

def read_and_process_dataset(dataset):
    ordered_special_words_list = []
    split_text_list = []
    
    for example in dataset:
        text = example['text']
        ordered_special_words, split_text = process_text(text)
        ordered_special_words_list.append(ordered_special_words)
        split_text_list.append(split_text)
    
    return ordered_special_words_list, split_text_list



def translate(data,save_dir,target_lang='az_Latn', NLLBversion=0):
    if NLLBversion==2:
        checkpoint = 'facebook/nllb-200-3.3B'
    if NLLBversion==0:
        checkpoint = 'facebook/nllb-200-1.3B'
    if NLLBversion==1:
        checkpoint = 'facebook/nllb-200-distilled-1.3B' 
    if NLLBversion==3:
        checkpoint = 'facebook/nllb-moe-54b'

    
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    translation_pipeline = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='en', tgt_lang=target_lang, max_length=1000)
    SWords,texts = read_and_process_dataset(data)
    translated_text=[]
    translated_texts=[]
    for text in texts:
        for i in text:
            output=translation_pipeline(i)
            translated_text.append(output[0]['translation_text'])
        translated_texts.append(translated_text)
    
    text=reconstruct_texts(SWords,translated_texts)

    for idx, content in enumerate(text):
        file_name = f"translated_sample_{idx}.txt"
        file_path = os.path.join(save_dir, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

    print("All files have been saved :D")


dir="C:\\Users\\quliy\\Desktop\\Wiki40\\translatedsamples"
translate(data,dir,NLLBversion=2)

