# Import libraries
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from transformers import set_seed
import joblib


# Set seed for reproducibility
SEED = 55
set_seed(SEED)

# Create empty DF
df = pd.DataFrame(columns=['id','findings','impression'])

# Register path to data folder
document_path = Path('../data/NLMCXR_reports/ecgen-radiology')

# Loop through all files in the folder
for doc in document_path.iterdir():

    # Check file ext
    if doc.suffix == '.xml':

        # Parse file
        medical_data = ET.parse(doc)
        medical_data_root = medical_data.getroot()

        # Extract id, findings & impression
        id_medical = medical_data_root.find('IUXRId').attrib.get('id')
        abstract_data = medical_data_root.find('.//Abstract')
        findings_data = abstract_data.find('.//AbstractText[@Label="FINDINGS"]').text
        impression_data = abstract_data.find('.//AbstractText[@Label="IMPRESSION"]').text

        # Append data to df
        df = pd.concat([df, pd.DataFrame({'id': [id_medical], 'findings': [findings_data], 'impression': [impression_data]})], ignore_index=True)

# Convert id column from string to int
df['id'] = df['id'].astype(int)

# Sort df by id & drop rows with missing cells
df = df.sort_values(by='id').reset_index(drop=True)
df.dropna(inplace=True)

# Regex pattern to change redacted PII with '[REDACTED]'
regex_pattern = r'[Xx]{2,}|[Xx]-X{2,}'

# Regex pattern to remove numbered lists 
regex_pattern_2 = r'\d+\.\s+'

# Regex pattern to remove duplicated periods
regex_pattern_3 = r'\s*\.\s*\.'

# Regex pattern to remove extra spaces
regex_pattern_4 = r'\s+'

# Apply regex patterns
df['findings'] = df['findings'].str.replace(regex_pattern, '[REDACTED]', regex=True).str.replace(regex_pattern_2, '', regex=True).str.replace(regex_pattern_3, '.', regex=True).str.replace(regex_pattern_4, ' ', regex=True)
df['impression'] = df['impression'].str.replace(regex_pattern, '[REDACTED]', regex=True).str.replace(regex_pattern_2, '', regex=True).str.replace(regex_pattern_3, '.', regex=True).str.replace(regex_pattern_4, ' ', regex=True)

# Drop duplicates where findings and impression are the same, keeping only one row
df = df.drop_duplicates(subset=['findings', 'impression'], keep='first')

# Export the cleaned dataframe to a CSV file
df.to_csv('../data/cleaned_data.csv', index=False)
print("Data cleaning complete. Data exported to '../data/cleaned_data.csv'.")


# Regex patterns to classify normal and abnormal cases
normal_case_regex = r'no acute|no evidence of|negative|clear lungs|normal|unremarkable'
abnormal_case_regex = r'atelectasis|effusion|cardiomegaly|congestion|pneumothorax'

# Categorize cases based on regex
case_filter = df['impression'].str.contains(normal_case_regex, case=False, regex=True) & ~df['impression'].str.contains(abnormal_case_regex, case=False, regex=True)
df['key'] = np.where(case_filter, 'normal', 'abnormal')


# Train Test Validation Split 80/10/10
train_df, val_test_df = train_test_split(df, test_size=0.2, stratify=df['key'], random_state=SEED)
val_df, test_df = train_test_split(val_test_df, test_size=0.5, stratify=val_test_df['key'], random_state=SEED)

# Load base model & tokenizer
print('Downloading model&tokenizer...')
model_name = 'google-t5/t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print('Model&tokenizer downloaded.')

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

# Model path
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
print('Starting training...') 
trainer.train()
print('Training completed.')

# Model save path
finetune_model_path = '../model/t5-summarization'

# Save fine tuned model
updated_model = model.merge_and_unload()
updated_model.save_pretrained(finetune_model_path)
tokenizer.save_pretrained(finetune_model_path)

print(f'Model saved to {finetune_model_path}')

print('Starting TF-IDF vectorization...') 
# Extract findings and convert to list
findings = df['findings'].tolist()

# Create vectorizer
# Define custom stopwords for the vectorizer
custom_stopwords = ['is', 'are', 'the', 'of', 'or', 'and', 'without', 'there', 'redacted', 'no', 'in']
# Ignore terms that appear more than 80%
vectorizer = TfidfVectorizer(max_df=.8, ngram_range=(1,2), stop_words=custom_stopwords)

# Fit&transform findings
tfidf_matrix = vectorizer.fit_transform(findings)

# Retrieve 100 top words
count_vectorizer = CountVectorizer(max_df=.9, min_df=2, stop_words=custom_stopwords)
count_matrix = count_vectorizer.fit_transform(findings)

word_counts = count_matrix.sum(axis=0).A1
words = count_vectorizer.get_feature_names_out()
word_freq = sorted(zip(words,word_counts), key=lambda x: x[1], reverse=True)
top_words = list(word for word, _ in word_freq[:100])

# TF-IDF model save path
tfidf_path = '../model/tfidf/'

# Save trained model 
joblib.dump(vectorizer, tfidf_path + 'tfidf_vectorizer.joblib')
joblib.dump(tfidf_matrix, tfidf_path + 'tfidf_matrix.joblib')
joblib.dump(top_words, tfidf_path + 'top_words.joblib')
print(f'TFIDF model saved to {tfidf_path}')