# Import data from kaggle
import kagglehub

print("Starting download of dataset...")
# Download latest version of the Vibrent Clothes Rental dataset
data_path = kagglehub.dataset_download("kaborg15/vibrent-clothes-rental-dataset")

print("Download completed. Dataset saved to:", data_path)

# Import libraries
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
import os
import shutil
from torchvision import models
import torch

# Save locally datasets
os.makedirs('data', exist_ok=True)
data1 = pd.read_csv(f'{data_path}/outfits.csv', sep=';')
data1.to_csv('data/outfits.csv')
print("Saving outfits.csv locally...")
data2 = pd.read_csv(f'{data_path}/picture_triplets.csv', sep=';')
data2.to_csv('data/picture_triplets.csv')
print("Saving picture_triplets.csv locally...")

# Download images to local environment
image_path = f'{data_path}/images'
download_path = 'data/images'
os.makedirs(download_path, exist_ok=True)

print("Starting image download...")
for image in data2['file_name']:
    original_img = f'{image_path}/{image}'
    shutil.copy(original_img, f'{download_path}/{image}')
print("All images copied successfully.")


# Load outfits and picture triplets csv and drop irrelevant columns
data_outfits = pd.read_csv('data/outfits.csv')
data_outfits = data_outfits[['id','group']]
print('Outfits data loaded successfully.')
print(data_outfits.info())
# Load picture triplets csv
data_picture = pd.read_csv('data/picture_triplets.csv')
data_picture.drop(columns=['displayOrder'], inplace=True)
print('Picture triplets data loaded successfully.')
print(data_picture.info())

# Drop rows with missing values in the outfits data
print('Dropping rows with missing values in the outfits data...')
data_outfits.dropna(inplace=True)

# Check for outfits without corresponding pictures
data_picture_null = data_outfits[~data_outfits['id'].isin(data_picture['outfit.id'])]
print('Number of outfits without corresponding pictures:', len(data_picture_null))

# Check for pictures without corresponding outfits
data_outfits_null = data_picture[~data_picture['outfit.id'].isin(data_outfits['id'])]
print('Number of pictures without corresponding outfits:', len(data_outfits_null))

# Remove any records without corresponding pictures
print('Removing records without corresponding pictures...')
data_outfits = data_outfits[data_outfits['id'].isin(data_picture['outfit.id'])]
print('Records without corresponding pictures removed.')
print(data_outfits.info())

# Remove any records without corresponding outfits in the picture triplets data
print('Removing records without corresponding outfits...')
data_picture = data_picture[data_picture['outfit.id'].isin(data_outfits['id'])]
print('Records without corresponding outfits removed.')
print(data_picture.info())

print(f'Length of data_outfits: {len(data_outfits)}')
print(f'Length of data_picture: {len(data_picture)}')

# Create image path and check if image exists
data_picture['image_path'] = data_picture['file_name'].apply(lambda x: f'data/images/{x}')
data_picture['image_exists'] = data_picture['image_path'].apply(lambda x: os.path.exists(x))

print('Number of images that exist:', data_picture['image_exists'].sum())
print('Number of images that do not exist:', len(data_picture) - data_picture['image_exists'].sum())

# Merge both dataframes 
merged_data = data_picture.merge(data_outfits, left_on='outfit.id', right_on='id', how='left').drop(columns=['id', 'image_exists'])

print(f"Merged data shape: {merged_data.shape} \n" )
print(f"Merged data missing values: \n{merged_data.isna().sum()}")


# Define function to verify images are not corrupted and meet minimum size requirements
print('Verifying images...')
def verify_image(image):
    try:
        with Image.open(image) as img:
            img.verify()
            w, h = img.size
            return w >= 100 and h >= 100
        return True
    except Exception:
        return False
    
merged_data['valid_image'] = merged_data['image_path'].apply(lambda x: verify_image(x))

print('Number of valid images:', merged_data['valid_image'].sum())
print('Number of invalid images:', len(merged_data) - merged_data['valid_image'].sum())

# Remove invalid images
merged_data = merged_data[merged_data['valid_image']==True]
print('Invalid images removed.')
merged_data['valid_image'].value_counts()
print(f"Remaining valid images: {merged_data['valid_image'].sum()}")
print(f"Remaining invalid images: {len(merged_data) - merged_data['valid_image'].sum()}")

# Define function to check for duplicate images using phash
def check_duplicates(image):
    return str(imagehash.phash(Image.open(image)))

print('Checking for duplicate images...')
merged_data['image_phash'] = merged_data['image_path'].apply(lambda x: check_duplicates(x))
print('Duplicate image check completed.')
print(f'Duplicated images: {merged_data["image_phash"].duplicated().sum()}')

# Drop duplicated images and remove unnecessary columns
print('Removing duplicated images...')
merged_data = merged_data.drop_duplicates(subset='image_phash', keep='first').reset_index(drop=True).drop(columns=['valid_image', 'image_phash'])

print(f'Remaining images after removing duplicates: {len(merged_data)}')

print('Data preprocessing completed.')

print('Load MobileNet model')

model_name = 'mobilenet_v3_large'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Set model and weights
weights = models.MobileNet_V3_Large_Weights.DEFAULT
model = models.mobilenet_v3_large(weights=weights)

# Replace classification layer with identity to avoid computing class scores and retrieve embeddings
model.classifier = torch.nn.Identity()

# Move to device and set to inference mode
model = model.to(device).eval()
preprocessor = weights.transforms()  

# Save path 
save_path = 'data/'

print('Starting embedding extraction...')
# Set empty list for embeddings, ID and invalid images
all_embeddings = []
all_ids = []
invalid = []
for _, row in merged_data.iterrows():
    try:
        image = Image.open(f'{row['image_path']}').convert('RGB')
    except Exception as e:
        print(f'Error with: {row["file_name"]}: {e}')
        invalid.append((row['picture.id'], row['file_name']))
        continue

    inputs = preprocessor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embeddings = model(inputs).squeeze().cpu().numpy()

    all_embeddings.append(embeddings)
    all_ids.append(row['picture.id'])

print(f'Total embeddings: {len(all_embeddings)}, Total IDs: {len(all_ids)}')
print(f'Total invalid images: {len(invalid)}')

os.makedirs(f'{save_path}/embeddings', exist_ok=True)
np.save(f'{save_path}/embeddings/embeddings.npy', np.array(all_embeddings))
np.save(f'{save_path}/embeddings/picture_ids.npy', np.array(all_ids))

print(f'Embeddings saved to: {save_path}/embeddings/embeddings.npy')
print(f'Picture IDs saved to: {save_path}/embeddings/picture_ids.npy')


# Create embeddings DF and merge it with merged_data
embeddings_df = pd.DataFrame({'picture.id': all_ids, 'embeddings_idx': range(len(all_embeddings))})
catalog = merged_data.merge(embeddings_df, on='picture.id', how='inner')

# Save DF as catalog.csv
print(f'Full_data: {len(merged_data)}')
print(f'Catalog: {len(catalog)}')
catalog.to_csv(f'{save_path}/catalog.csv', index=False)
print(f'Catalog saved to: {save_path}/catalog.csv')


# Save final images 
os.makedirs('data/final_images', exist_ok=True)

print("Starting image copy...")
for image in catalog['file_name']:
    original_img = f'data/images/{image}'
    shutil.copy(original_img, f'data/final_images/{image}')
print("All images copied successfully.")
