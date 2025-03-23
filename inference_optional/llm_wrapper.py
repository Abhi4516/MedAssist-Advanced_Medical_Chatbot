# optional 

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
import yaml
from custom_logging.logger import get_logger

logger = get_logger("inference_log")

class LLMWrapper:
    def __init__(self, model_dir, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.model.to(self.device)
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1
        )
    
    def generate_response(self, instruction, input_text=""):
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True
        )
        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        logger.info("Generated response: " + response)
        return response

if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_dir = config['training']['output_dir']
    llm = LLMWrapper(model_dir)
    test_response = llm.generate_response("If you are a doctor, please answer the medical query.", "I have a headache.")
    print("Response:", test_response)
