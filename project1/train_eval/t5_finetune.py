# Import libraries
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, GenerationConfig, set_seed
import torch
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Data path
data = '../data/cleaned_data.csv'

# Set seed for reproducibility
SEED = 55
set_seed(SEED)

# Load data to DF
df = pd.read_csv(data)

# Regex patterns to classify normal and abnormal cases
normal_case_regex = r'no acute|no evidence of|negative|clear lungs|normal|unremarkable'
abnormal_case_regex = r'atelectasis|effusion|cardiomegaly|congestion|pneumothorax'

# Categorize cases based on regex
case_filter = df['impression'].str.contains(normal_case_regex, case=False, regex=True) & ~df['impression'].str.contains(abnormal_case_regex, case=False, regex=True)
df['key'] = np.where(case_filter, 'normal', 'abnormal')


# Train Test Validation Split 80/10/10
train_df, val_test_df = train_test_split(df, test_size=0.2, stratify=df['key'], random_state=SEED)
val_df, test_df = train_test_split(val_test_df, test_size=0.5, stratify=val_test_df['key'], random_state=SEED)

# Export train/val/test datasets to csv
train_df.to_csv('../data/train_data.csv', index=False)
val_df.to_csv('../data/val_data.csv', index=False)
test_df.to_csv('../data/test_data.csv', index=False)

# Load base model & tokenizer
model_name = 'google-t5/t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Add special token for [REDACTED]
tokenizer.add_special_tokens({'additional_special_tokens': ['[REDACTED]']})
model.resize_token_embeddings(len(tokenizer))

# Tokenize data function
def tokenize_data(data):
    model_inputs = tokenizer(data['findings'], max_length=512,truncation=True)
    labels = tokenizer(data['impression'], max_length=512, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Convert DF to dataset
train_dataset = Dataset.from_pandas(train_df.drop(columns=['key']), preserve_index=False)
test_dataset = Dataset.from_pandas(test_df.drop(columns=['key']), preserve_index=False)
val_dataset = Dataset.from_pandas(val_df.drop(columns=['key']), preserve_index=False)

# Tokenize dataset
train_dataset_tokenized = train_dataset.map(tokenize_data, batched=True, remove_columns=train_dataset.column_names)
test_dataset_tokenized = test_dataset.map(tokenize_data, batched=True, remove_columns=test_dataset.column_names)
val_dataset_tokenized = val_dataset.map(tokenize_data, batched=True, remove_columns=val_dataset.column_names)

# Define lora config
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=['q','k','v','o'],
    lora_dropout=0.5,
    bias='none',
    task_type=TaskType.SEQ_2_SEQ_LM
)

# Apply lora config to model
model = get_peft_model(model, lora_config)

# Define generation config
gen_conf = GenerationConfig(
    max_new_tokens=128,
    num_beams=4,
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,
    early_stopping=True
)

# Local path
model_path = '../models/t5'

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=model_path,
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=15,
    logging_steps=10,
    seed=SEED,
    load_best_model_at_end=True
)

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tokenized,
    eval_dataset=val_dataset_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator
)

# Start training    
trainer.train()

# Local save path
finetune_model_path = '../model/t5-summarization'

# Save fine tuned model
updated_model = model.merge_and_unload()
updated_model.save_pretrained(finetune_model_path)
tokenizer.save_pretrained(finetune_model_path)
