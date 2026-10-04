## This script move the images into subdirectories, each containing up to 5k images ##
## Hugging Face Dataset is limited to 10k files per directory ##
## Images are hosted in https://huggingface.co/datasets/Keizano/images ##

# Import libraries
import shutil
from pathlib import Path
import pandas as pd

# Load the catalog of image file names
catalog = pd.read_csv('data/catalog.csv')

# Path to images
source = 'data/copied_images'

# Loop through catalog and move images to subdirectories
for run, img_name in enumerate(catalog['file_name']):
    run_index = run // 5000 + 1
    export = Path(f'data/images/sub_{run_index}')
    export.mkdir(parents=True, exist_ok=True)

    src = f'{source}/{img_name}'
    dst = f'{export}/{img_name}'

    shutil.move(str(src), str(dst))
shutil.rmtree(source)