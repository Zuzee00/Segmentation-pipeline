import json

from split_data_to_train_val_test import SplitData


with open('config.json', 'r') as file:
    config = json.load(file)

# Step 1
split_data = SplitData(root_dir=config['root_dir'], images_path=config['images_path'], masks_path=config['masks_path'],
                       folder_names=config['folders'])
split_data.create_train_val_test_sets()
