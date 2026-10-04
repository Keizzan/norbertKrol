# Visual Product Recommendations

Upload a photo of a clothing item and get the 5 most visually similar products from the retailer's catalog, each with its cosine similarity score.

**Live app:** URL

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
| `data/catalog.csv` | Catalog of cleaned images with embedding indices |
| `data/embeddings/` | Saved catalog embeddings and picture IDs |
| `data/test_images/` | Test images used for the sample outputs |

## Reproducing the project

1. Install the dependencies: `pip install -r requirements.txt`
2. Set up Kaggle access for `kagglehub` (a Kaggle account is required).
3. Run `data_pipeline.py` to download and clean the data, extract embeddings, and write `data/catalog.csv` and `data/embeddings/`.
   - Alternatively, run `colab_first_proper.ipynb` on Google Colab to use a GPU.
4. Run `rest_images.py` to split the images into subfolders.
5. Open `onepointtwo.ipynb` to reproduce the test outputs and plots.
6. Run the app locally: `streamlit run app.py`

## Deployment

The app is deployed on Render. It loads `data/catalog.csv` and `data/embeddings/embeddings.npy` from the repository and fetches the catalog images from Hugging Face. The deployed code is `app.py`, with dependencies from `requirements.txt`.