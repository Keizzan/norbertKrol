#### Model Summary
This project implements summarization system using fine tuned T5-base model. 'FINDINGS'(input) and 'IMPRESSION'(output) was extracted from XML files that contains medical reports. EDA performed on data includes removing empty fields and duplicated records, removing duplicated punctuation, list formatting and standardize PPI. Processed data was extraced to 'cleaned_data.csv'. Model was trained with LoRa to reduce trainable parameters, which helps trainig process be quicker and reduce memory usage. LoRa affect query, key, value and output modules. Dataset was split into training, validation and test sets in 8:1:1 ratio. Split was stratified based on normal and abnormal case labels. Model was trained on Google Colab GPU L4. Training rate was 5e-5 with 15 epochs and batch size of 16. Rouge1 F1 score on test set was 0.38 and validation set 0.32. Final system contains fined tuned model and TF-IDF vectorizer working as gatekeeper to filter users input, and rejecting irrelevant requests. 

#### Limitations and recommendations
Implementation is limited by dataset and training approach. LoRa reduce computations cost, so model can be trained locally, but full fine tune would produce bigger impact on model perfomance. Also in feature creation, key (normal/abnormal labels) were created, by using most common phrases appearing in dataset + most common words appearing in chest xrays datasets, manual labeling would improve quality. In addition TF-IDF validation step is heuristic filter and may reject some valid, but rare inputs. To improve system perfomance, model could be trained on bigger dataset or use larger base model.

#### REST API documentation
This api accepts radiology reports in json format with 'input' as a key and returns generated summary also in json with repeated 'input' and 'output'.

URL: https://norbertkrol.onrender.com/
Method: POST
Authentication required: No

Data example
{'input':'The trachea is midline. The cardiomediastinal silhouette is normal. The lungs are clear, without evidence of acute infiltrate or effusion. There is no pneumothorax. The visualized bony structures reveal no acute abnormalities.'}

Success Response
Code: 200
Content example:
{'input': 'The trachea is midline. The cardiomediastinal silhouette is normal. The lungs are clear, without evidence of acute infiltrate or effusion. There is no pneumothorax. The visualized bony structures reveal no acute abnormalities.', 'output':'No acute cardiopulmonary abnormalities.'}

Error response
Condition: Input too short or too long, doesn't contain medical vocabulary or doesn't resemble medical report
Code: 400
Content:{'input': 'This recepie requires flour, sugar and eggs. It is very quick and easy.', 'output': 'Your input is not suitable for this function'}