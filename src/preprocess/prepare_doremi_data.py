from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps

DOREMI_IMAGES_ROOT_DIR = Path("data/DoReMi_v1/DoReMi_v1/Images")
NEW_DIR_PATH = Path("data/doReMiPrepared")
# this script extracts all the images from the DoReMi dataset 
# and also (very roughly) crops them to only contain the singular system

NEW_DIR_PATH.mkdir(exist_ok=True)
img_dir = NEW_DIR_PATH / "images"
img_dir.mkdir(exist_ok=True)
id_list = []
DEFAULT_CROP = (0, 200, 0, 2500)
# images that are of single-system scores not by actual composers are labeled with one of these tags
NON_COMPOSER_TAGS = ["accidental", "beam", "syncopation"]
TRIO_CROP = (0, 200, 0, 2000)
QUARTET_CROP = (0, 200, 0, 1650)
PIANO_CROP = (0, 200, 0, 2200)

for item in tqdm(DOREMI_IMAGES_ROOT_DIR.iterdir()):
    if item.suffix != ".png": # skip non-image metadata file
        continue
    img_id = item.stem
    id_list.append(img_id)
    img = Image.open(DOREMI_IMAGES_ROOT_DIR / (img_id + ".png"))
    crop_border = DEFAULT_CROP
    # if the image is of an actual work, we may need to adjust the cropping
    if not any(tag in img_id for tag in NON_COMPOSER_TAGS):
        if any(tag in img_id.lower() for tag in ["trio", "mikrokosmos", "nights music", "solo violin sonata"]):
            crop_border = TRIO_CROP
        elif any(tag in img_id.lower() for tag in ["quartet", "reger - introduction"]):
            crop_border = QUARTET_CROP
        elif any(tag in img_id.lower() for tag in ["piano", "alkan", "variation", "scriabin", "beethoven", "chopin", "janacek", "mendelssohn", "reger - improv"]):
            crop_border = PIANO_CROP

    img = ImageOps.crop(img, crop_border)
    img.save(img_dir / (item.stem + ".png"))

id_df = pd.DataFrame({"id": id_list})
id_df.to_csv(NEW_DIR_PATH / "ids.csv")