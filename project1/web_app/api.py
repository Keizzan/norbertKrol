# Import libraries
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
import joblib
import json
from sklearn.metrics.pairwise import cosine_similarity

# Set device
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
# Model path
model_path = "model/t5-summarization"

# Load tokenizer&model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

# Load tfidf
vectorizer = joblib.load('model/tfidf/tfidf_vectorizer.joblib')
tfidf_matrix = joblib.load('model/tfidf/tfidf_matrix.joblib')
top_words = joblib.load('model/tfidf/top_words.joblib')

# Define generation config
gen_conf = GenerationConfig(
    max_new_tokens=128,
    num_beams=4,
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,
    early_stopping=True
)

# Check similarity score on input
def check_similarity(input):
    input_vec = vectorizer.transform([input])
    similarity = cosine_similarity(input_vec, tfidf_matrix).flatten()
    return similarity.max()

# Check if vocab is present in the input
def vocab_presence(input, vocab=top_words):
    if [i for i in vocab if i in input.lower()]:
        return True
    return False

# Check input validity
def validate_input(input):
    if not isinstance(input, str):
        return False, 'Incorrect input. Input must be a string.'
    if len(input.strip()) < 10:
        return False, 'Input is too short.'
    if len(input.strip()) > 2000:
        return False, 'Input is too long. Max is 2000 characters.'
    if vocab_presence(input):
        if check_similarity(input) > 0.2:
            return True, 'Validation successful'
        else:
            return False, 'Your input does not resemble radiology report'
    else:
        return False, 'Your input is not suitable for this function'

def summarize(text):
    status, message = validate_input(text)
    if not status:
        return message, 400
    else:
        inputs = tokenizer(text, return_tensors='pt', truncation=True).to(device)

        with torch.no_grad():
            output = model.generate(**inputs, generation_config=gen_conf)
        return tokenizer.decode(output[0], skip_special_tokens=True), 200

# Load score results from json
with open('data/results.json') as f:
    scores = json.load(f)

# Flask App
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    error = None
    result = None
    if request.method == 'POST':
        input = request.form.get('input_text')
        output, status = summarize(input)
        if status == 200:
            result = {'input': input, 'output': output}
        else:
            error = output
    return render_template('index.html', scores=scores, result=result, error=error)

@app.route('/api_endpoint', methods=['POST'])
def api_endpoint():
    input = request.get_json()['input']
    output, status = summarize(input)
    return jsonify({'input': input, 'output': output}), status
if __name__ == "__main__":
    app.run(debug=True)
