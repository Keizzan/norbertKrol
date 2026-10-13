# Visual Product Recommendations

Upload a photo of a clothing item and get the 5 most visually similar products from the retailer's catalog, each with its cosine similarity score.

**Live app:** https://norbertkrol1.onrender.com/

## Model summary

The Vibrent Clothes Rental Dataset was cleaned by dropping outfits with missing data, pictures without a matching outfit (and vice versa), corrupted or undersized images (< 100 px), and pHash duplicates. This left 30,366 unique images.

Each image is vectorized with a pre-trained MobileNetV3-Large. The classifier head is replaced by an identity layer, so the model outputs embeddings rather than classification. No fine-tuning was done.

All catalog embeddings are computed once and saved (`data/embeddings/embeddings.npy`, with `data/catalog.csv` as the index). At query time the uploaded image is embedded the same way, and its cosine similarity to every catalog embedding is computed. The top 5 results are returned, keeping only one item per outfit group to avoid near-identical results. The catalog images are hosted on Hugging Face (`Keizano/images`), and the app is built with Streamlit.

**Sample output:**

| Test image | Top-5 cosine similarity scores |
|---|---|
| `test_img_1.jpg` | 0.924, 0.882, 0.880, 0.878, 0.876 |
| `test_img_2.jpg` | 0.805, 0.803, 0.802, 0.799, 0.799 |

All scores are above the 0.7 target.

## Limitations and future improvements

- The model is a generic ImageNet network and was not fine-tuned on fashion data, so it can match on colour or pattern rather than garment type. Fine-tuning, or a fashion-aware model such as CLIP or a vision transformer, would improve relevance.
- The one-result-per-group filter helps diversity but can hide a closer match. Metadata such as category, colour and price is not used for filtering.
- Images are fetched from a remote host, so load time depends on the connection.

## Project structure

| File | Purpose |
|---|---|
| `data_pipeline.py` | Data download, cleaning, embedding extraction, and `catalog.csv` creation |
| `first_notebook.ipynb` | Colab version of the pipeline, used for GPU embedding extraction |
| `second_notebook.ipynb` | Output functions (cosine similarity, top-N) and test image results |
| `app.py` | Streamlit web app (deployed) |
| `rest_images.py` | Splits images into 5,000-image subfolders for Hugging Face hosting |
| `requirements.txt` | Python dependencies |

## Requirements

- Python **3.12** (tested with 3.12.3 on Linux). `data_pipeline.py` uses nested same-type quotes inside an f-string, which needs 3.12 or newer.
- A Kaggle account for `kagglehub` (used to download the dataset).
- Optional: a CUDA GPU. The pipeline falls back to CPU, which is slow for about 30k images.

## Installation

Run these from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data directory structure

```text
data/
    outfits.csv            # from Kaggle: kaborg15/vibrent-clothes-rental-dataset
    picture_triplets.csv   # from Kaggle: kaborg15/vibrent-clothes-rental-dataset
    catalog.csv            # cleaned catalog (30,366 images) with embeddings_idx
    embeddings/
      embeddings.npy     # one MobileNetV3-Large vector per catalog image
      picture_ids.npy    # picture.id for each embedding row
    images/                # created by data_pipeline.py (raw images, not committed)
    final_images/          # created by data_pipeline.py (catalog images, not committed)
    test_images/              
      test_img_1.jpg
      test_img_2.jpg
```

`app.py` only needs `data/catalog.csv` and `data/embeddings/embeddings.npy`. Catalog images are loaded from `Keizano/images` on Hugging Face.

## Running locally

With the prebuilt data already in the repository:

```bash
streamlit run app.py
```

Open http://localhost:8501, upload a `.png`, `.jpg` or `.jpeg` image, and click **Recommend**.

## Reproducing the project

All commands run from the repository root, because the scripts use relative `data/` paths.

1. Install the dependencies from `requirements.txt`.
2. Configure Kaggle credentials for `kagglehub`.
3. Build the data. This downloads the dataset, cleans it, extracts embeddings, and writes `data/catalog.csv` and `data/embeddings/`:
   ```bash
   python3 data_pipeline.py
   ```
4. *(Optional, only for hosting images on Hugging Face)* Split the images into 5,000-image subfolders:
   ```bash
   python3 rest_images.py
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```

## Deployment

The app is deployed on Render. It loads `data/catalog.csv` and `data/embeddings/embeddings.npy` from the repository and fetches the catalog images from Hugging Face. The deployed code is `app.py`, with dependencies from `requirements.txt`.