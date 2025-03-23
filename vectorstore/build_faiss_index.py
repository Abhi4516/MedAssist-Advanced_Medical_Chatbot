

import pandas as pd
import yaml
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from custom_logging.logger import get_logger

logger = get_logger("vector_base_log")

def build_faiss_index(config_path="config.yaml", sample_size=700000, output_path="faiss_index2"):
    """
    Reads the dataset (with 'input' and 'output' fields), takes a random sample,
    and builds a FAISS index from the text embeddings.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dataset_path = config["dataset"]["path"]
    logger.info(f"Reading dataset from {dataset_path}...")

    try:
        df = pd.read_json(dataset_path)
    except Exception as e:
        logger.error(f"Failed to read dataset: {e}")
        raise e

    total_len = len(df)
    logger.info(f"Total dataset size: {total_len}")

    sample_size = min(sample_size, total_len)
    df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    logger.info(f"Using {sample_size} rows from the dataset to build FAISS index.")

    texts = []
    for idx, row in df_sample.iterrows():
    
        inp_val = (row.get("input") or "").strip().lower()
        out_val = (row.get("output") or "").strip().lower()

        combined_text = f"Input: {inp_val}\nOutput: {out_val}"
        texts.append(combined_text)

    logger.info("Successfully created text representations (only 'input' and 'output').")

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Embeddings created using 'sentence-transformers/all-MiniLM-L6-v2'.")
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise e

    try:
        vector_store = FAISS.from_texts(texts, embeddings)
        logger.info("FAISS vector store built successfully.")
    except Exception as e:
        logger.error(f"Error building FAISS vector store: {e}")
        raise e

    try:
        vector_store.save_local(output_path)
        logger.info(f"FAISS index saved to '{output_path}'.")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")
        raise e

if __name__ == "__main__":
    build_faiss_index()
