
from transformers import AutoTokenizer
import pandas as pd
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def preprocess_dataframe(df, tokenizer_name, max_length=512):
    """
    Combine 'instruction', 'input', and 'output' fields into a single text prompt.
    Then tokenize the texts.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def combine_fields(row):   
        instruction = row.get('instruction', '')
        input_text = row.get('input', '')
        output = row.get('output', '')
        combined = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
        return combined
    
    df['combined'] = df.apply(combine_fields, axis=1)
    tokenized_data = tokenizer(
        df['combined'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_length
    )
    return tokenized_data

if __name__ == "__main__":
    config = load_config()
    dataset_path = config['dataset']['path']
    df = pd.read_json(dataset_path, lines=True)
    tokenized_data = preprocess_dataframe(df, config['model']['tokenizer_name'])
    print("Tokenized data keys:", tokenized_data.keys())
