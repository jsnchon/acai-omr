from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm

PRIMUS_ROOT_DIR = Path("data/primusCalvoRizoAppliedSciences2018")
NEW_DIR_PATH = Path("data/primusPrepared")
# Primus dataset contains a directory for each image with a lot of files. This script just extracts
# the pngs and also creates a csv containing all their ids for use in a dataset object

NEW_DIR_PATH.mkdir(exist_ok=True)
new_img_dir = NEW_DIR_PATH / "images"
new_img_dir.mkdir(exist_ok=True)
id_list = []
for package in PRIMUS_ROOT_DIR.iterdir():
    print(f"Extracting from {package}")
    for example_dir in tqdm(package.iterdir()):
        id_list.append(example_dir.stem)
        img_path = example_dir / (example_dir.stem + ".png")
        shutil.copy(img_path, new_img_dir)

id_df = pd.DataFrame({"id": id_list})
id_df.to_csv(NEW_DIR_PATH / "ids.csv")