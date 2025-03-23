

import yaml
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from data.data_handler import load_config, load_dataset, clean_dataset, split_dataset
from custom_logging.logger import get_logger


logger = get_logger("training_log")

def freeze_layers(model, freeze_encoder_layers=4, freeze_decoder_layers=4):
    """
    Freeze the lower layers of both encoder and decoder.
    For FLAN-T5 Small (assumed ~6 layers each), we freeze the first few layers.
    """
    for i, layer in enumerate(model.encoder.block):
        if i < freeze_encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
    for i, layer in enumerate(model.decoder.block):
        if i < freeze_decoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
    return model

class EpochLoggerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.log_history:
            latest_metrics = state.log_history[-1]
            logger.info(f"Epoch {state.epoch} ended with metrics: {latest_metrics}")
        else:
            logger.info(f"Epoch {state.epoch} ended, no metrics found.")

def main():
    config = load_config()
    

    dataset_path = config['dataset']['path']
    df = load_dataset(dataset_path)
    df = clean_dataset(df)
    df = df.sample(n=20000, random_state=42)  # Subsample 20k rows for training
    
    train_df, val_df, _ = split_dataset(df)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    if config['training'].get("use_peft", False):
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            inference_mode=False,
            r=config['training'].get("lora_r", 8),
            lora_alpha=config['training'].get("lora_alpha", 32),
            lora_dropout=config['training'].get("lora_dropout", 0.1),
            target_modules=config['training'].get("target_modules", ["q", "v"])
        )
        model = get_peft_model(model, lora_config)
        logger.info("LoRA adapters added to the model. Only adapter parameters will be updated.")
    else:
        model = freeze_layers(
            model,
            freeze_encoder_layers=config['training'].get('freeze_encoder_layers', 4),
            freeze_decoder_layers=config['training'].get('freeze_decoder_layers', 4)
        )
        logger.info("Lower layers frozen. Only top layers are trainable.")
    
    def preprocess_function(examples):
        source_texts = []
        target_texts = []
        for instr, inp, out in zip(examples['instruction'], examples['input'], examples['output']):
            source = f"Instruction: {instr}\nInput: {inp}"
            target = out
            source_texts.append(source)
            target_texts.append(target)
        
        model_inputs = tokenizer(source_texts, truncation=True, padding="max_length", max_length=config['training']['max_length'])
        labels = tokenizer(text_target=target_texts, truncation=True, padding="max_length", max_length=config['training']['max_length']).input_ids
        
        labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
        model_inputs["labels"] = labels
        return model_inputs

    train_dataset = train_dataset.map(
        preprocess_function, batched=True, num_proc=4,
        remove_columns=train_dataset.column_names, load_from_cache_file=True
    )
    val_dataset = val_dataset.map(
        preprocess_function, batched=True, num_proc=4,
        remove_columns=val_dataset.column_names, load_from_cache_file=True
    )
    
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        eval_strategy="epoch",    
        save_strategy="epoch",
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=0.01,
        logging_dir=config['training']['logging_dir'],
        logging_steps=config['training']['logging_steps'],
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EpochLoggerCallback()]
    )
    
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")
    
    eval_metrics = trainer.evaluate()
    logger.info("Evaluation Metrics: " + str(eval_metrics))
    
    model.save_pretrained(config['training']['output_dir'])
    tokenizer.save_pretrained(config['training']['output_dir'])

if __name__ == "__main__":
    main()
