# split the COCO json for all labeled examples in the system detection dataset into train, test, and validation splits.
# Define which scores should be in validation and test and the rest will be put into train
import json
import re
from pathlib import Path

DATASET_ROOT_DIR = Path("data/system_detection")
WHOLE_JSON_PATH = DATASET_ROOT_DIR / "system_detection.json"
VALIDATION_SCORES = ["chopin_ballade_no_4", "debussy_reflets_dans_leau"]
TEST_SCORES = ["rach_liebesleid", "ravel_pavane"]

def split_json_from_ids(whole_json: dict, ids: list, save_path: str):
    split_images = [img for img in whole_json["images"] if img["id"] in ids]
    split_annotations = [annotation for annotation in whole_json["annotations"] if annotation["image_id"] in ids]
    split_dict = {"images": split_images, "categories": whole_json["categories"], "annotations": split_annotations}
    with open(save_path, "w") as f:
        print(f"Writing split file to {save_path}")
        indent_level = 2
        json.dump(split_dict, f, indent=indent_level)

with open(WHOLE_JSON_PATH, "r") as f:
    whole_json = json.load(f)

train_ids = []
validation_ids = []
test_ids = []

validation_pattern = "|".join(VALIDATION_SCORES)
test_pattern = "|".join(TEST_SCORES)

for img in whole_json["images"]:
    id = img["id"]
    file_name = img["file_name"]
    if re.search(validation_pattern, file_name):
        validation_ids.append(id)
    elif re.search(test_pattern, file_name):
        test_ids.append(id)
    else:
        train_ids.append(id)
    # label studio by default uses weird relative paths pointing to its uploads directory
    img["file_name"] = str(DATASET_ROOT_DIR / "images" / Path(file_name).name)
    
split_json_from_ids(whole_json, train_ids, "train.json")
split_json_from_ids(whole_json, validation_ids, "validation.json")
split_json_from_ids(whole_json, test_ids, "test.json")
