# for demo

from inference_optional.llm_wrapper import LLMWrapper
import yaml

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_dir = config['training']['output_dir']
    llm = LLMWrapper(model_dir)
    instruction = "If you are a doctor, please answer the medical query."
    input_text = "I have severe chest pain."
    response = llm.generate_response(instruction, input_text)
    print("Response:", response)

if __name__ == "__main__":
    main()
