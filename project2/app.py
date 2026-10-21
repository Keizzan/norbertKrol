import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity

img_path = 'https://huggingface.co/datasets/Keizano/images/resolve/main'

# Define cached resources for app
@st.cache_resource
def inital_load():
    # Load model and preprocessor
    device ='cpu'
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)
    model.classifier = torch.nn.Identity()
    model = model.to(device).eval()
    preprocessor = weights.transforms()

    # Load data
    catalog = pd.read_csv('data/catalog.csv')
    embeddings = np.load('data/embeddings/embeddings.npy')
    id_to_group = catalog.set_index('embeddings_idx')['group'].to_dict()
    return model, preprocessor, device, catalog, embeddings, id_to_group

model, preprocessor, device, catalog, embeddings, id_to_group = inital_load()

# Function to get image embedding
def get_embedding(image):
    img = Image.open(image).convert('RGB')
    input = preprocessor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(input).squeeze().cpu().numpy()



# Function to get top 5 similar items based on embedding
def get_top_5_similar(embedding, all_embeddings, id_to_group):
    # Compute similarity between input embedding and all embeddings in the catalog
    # And sort them from highest score
    similarity = cosine_similarity(embedding.reshape(1,-1), all_embeddings)[0]
    top_id = np.argsort(similarity)[::-1]

    # Set variables to track gropus, ids and scores
    groups = set()
    id, score = [],[]

    # Loop through highest scores items and keep only one item per group
    # Break loop at 5 items
    for i in top_id:
        group = id_to_group[i]
        if group in groups:
            continue
        groups.add(group)
        id.append(i)
        score.append(similarity[i])
        if len(id) == 5:
            break

    return id, score

# Images are hosted on HF dataset broken down into subfolders of 5000 images each
# Function to get image URL based on embedding index
def get_image_url(file):
    subfolder = file['embeddings_idx'] // 5000+1
    return f'{img_path}/sub_{subfolder}/{file["file_name"]}'


# Webapp Page
st.set_page_config(page_title='Visual Search Recommendation', page_icon='favicon-32x32.png')
st.title('Visual Search Recommendation')
st.write('Upload an image to get 5 recommended similar items')


uploaded_file = st.file_uploader('Upload image', type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    st.image(uploaded_file, caption='Uploaded Image', width=200)


    if st.button('Recommend'):
        embedding = get_embedding(uploaded_file)
        top_id, top_score = get_top_5_similar(embedding, embeddings, id_to_group)

        col = st.columns(5)
        for c, id, score in zip(col, top_id, top_score):
            row = catalog[catalog['embeddings_idx'] == id].iloc[0]
            c.write(f"Similarity score: {score:.3f}")
            c.image(get_image_url(row))