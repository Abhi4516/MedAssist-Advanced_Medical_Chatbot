
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def load_dataset(file_path):
    """
    Load a newline-delimited JSON dataset.
    Each line in the file should be a valid JSON object.
    """
    try:
        df = pd.read_json(file_path)
    except ValueError as e:
        raise ValueError(f"Error reading JSON file at {file_path}: {e}")
    return df

def clean_dataset(df):
    """
    Remove duplicates and rows with missing critical fields.
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=['instruction', 'output'])
    return df

def split_dataset(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Split the dataset into training, validation, and test sets.
    """
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio/(val_ratio+test_ratio), random_state=random_state)
    return train_df, val_df, test_df

if __name__ == "__main__":
    config = load_config()
    dataset_path = config['dataset']['path']
    df = load_dataset(dataset_path)
    df = clean_dataset(df)
    train_df, val_df, test_df = split_dataset(df)
    print("Train size:", len(train_df))
    print("Validation size:", len(val_df))
    print("Test size:", len(test_df))
