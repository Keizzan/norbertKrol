# Project 1 – Radiology Report Summarisation (FINDINGS → IMPRESSION)

Fine-tuned T5-base (LoRA) generates the IMPRESSION of a chest X-ray report from its FINDINGS. A TF-IDF gatekeeper rejects non-radiology input. Served via a Flask REST API.

## Setup
1. **Raw data:** extract the NLM Open-i `NLMCXR_reports` files are at `project1/data/NLMCXR_reports/ecgen-radiology/*.xml`.
2. **Install** (Python 3.12, from `project1/`; GPU recommended, first run downloads `google-t5/t5-base`):
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

## Run the pipeline
From `project1/`: `python run_pipeline.py`. Stages run in this order:

| # | Script | Output |
|---|---|---|
| 1 | `eda/eda.py` | `data/cleaned_data.csv` |
| 2 | `train_eval/t5_finetune.py` | `data/{train,val,test}_data.csv`, final model in `model/t5-summarization/` |
| 3 | `train_eval/tfidf.py` | `model/tfidf/*.joblib` |

Notebooks are development evidence only and are not used by the pipeline.

## Dataset cleaning statistics (printed by the data-processing stage)
Files processed 3955 · Records extracted 3955 · Missing FINDINGS 530 · Missing IMPRESSION 34 · Records removed (missing) 536 · Duplicates removed 724 · **Final cleaned size 2695** · Train/Val/Test **2156 / 269 / 270** (80/10/10, stratified).

## Model and results
T5-base + LoRA (r=64, alpha=128, targets q/k/v/o), lr 5e-5, 15 epochs, batch 16, seed 55. Gatekeeper: 10–2000 chars, top-100 TF-IDF word present, cosine similarity > 0.2.

| Split | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|---|---|---|---|
| Validation | 0.336 | 0.219 | 0.329 |
| **Test** | **0.382** | 0.243 | 0.377 |

Sample predictions:
- "The lungs are clear. Heart size and mediastinal contours are normal. No osseous abnormalities." → *No acute cardiopulmonary findings.*
- "AP and lateral views were obtained. Bibasilar atelectasis and small left-sided pleural effusion. Stable cardiomegaly. No pneumothorax. Mild pulmonary vascular congestion" → *Bibasilar atelectasis and small left-sided pleural effusion. Stable cardiomegaly.*

## REST API
**Live:** `POST https://norbertkrol.onrender.com/api_endpoint` (`Content-Type: application/json`)

- Request: `{"findings": "The lungs are clear. Heart size and mediastinal contours are normal. No osseous abnormalities."}`
- Success (200): `{"impression": "No acute cardiopulmonary findings."}`
- Error (400, missing/non-string/too short or long/non-radiology `findings`): `{"error": "Incorrect input. Input must be a string."}`

**Run locally** (after the pipeline, from `project1/`):
```bash
python web_app/api.py 
```
A browser form with ROUGE scores is at `http://127.0.0.1:5000/`.

## Limitations
LoRA instead of full fine-tuning; normal/abnormal labels are keyword-based; the TF-IDF filter is heuristic and may reject rare valid reports. Improvements: longer training, more data, larger base model.