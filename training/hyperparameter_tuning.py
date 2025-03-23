

import optuna
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from data.data_handler import load_config, load_dataset, clean_dataset, split_dataset
from custom_logging.logger import get_logger

logger = get_logger("hyperparameter_log")

def objective(trial):
    config = load_config()
    model_name = config['model']['name']
    
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-4)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    epochs = trial.suggest_int("epochs", 1, 3)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    df = load_dataset(config['dataset']['path'])
    df = clean_dataset(df)
    train_df, val_df, _ = split_dataset(df)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
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
    
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)
    
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'] + "/optuna_trial",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    logger.info("Trial finished with eval loss: " + str(eval_results["eval_loss"]))
    return eval_results["eval_loss"]

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_trial.params
    logger.info("Best hyperparameters: " + str(best_params))
    with open("best_hyperparams.yaml", "w", encoding="utf-8") as f:
        yaml.dump(best_params, f)

if __name__ == "__main__":
    main()
