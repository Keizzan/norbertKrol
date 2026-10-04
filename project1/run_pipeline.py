import os

print('Starting Pipeline')

print('Running EDA')
os.chdir('eda')
os.system('python3 eda.py')
print('EDA Done')

print('Finetune Model')
os.chdir('../train_eval')
os.system('python3 train_eval.py')
print('Finetune Model Done')

print('TfIdf Vectorization')
os.system('python3 tfidf.py')
print('TfIdf Vectorization Done')

print('Pipeline Done')