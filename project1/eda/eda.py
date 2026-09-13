# Import libraries
import pandas as pd
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from transformers import set_seed


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

