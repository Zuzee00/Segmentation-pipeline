import os
import random
import re
from PIL import Image

DATA_PATH = 'images_nearmap/'
FRAME_PATH = DATA_PATH + 'images'
MASK_PATH = DATA_PATH + 'annotations/binary'

# Create folders to hold images and masks

folders = ['train_frames', 'train_masks', 'val_frames', 'val_masks', 'test_frames', 'test_masks']

for folder in folders:
    if not os.path.exists(DATA_PATH + folder):
        os.makedirs(DATA_PATH + folder)

# Get all frames and masks, sort them, shuffle them to generate data sets.

all_frames = os.listdir(FRAME_PATH)
all_masks = os.listdir(MASK_PATH)

all_frames.sort(key=lambda var: [x if x.isdigit() else x
                                 for x in re.findall(r'[^0-9]|[0-9]+', var)])
all_masks.sort(key=lambda var: [x if x.isdigit() else x
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])

random.seed(230)
random.shuffle(all_frames)

# Generate train, val, and test sets for frames

train_split = int(0.7 * len(all_frames))
val_split = int(0.9 * len(all_frames))

train_frames = all_frames[:train_split]
val_frames = all_frames[train_split:val_split]
test_frames = all_frames[val_split:]

# Generate corresponding mask lists for masks

train_masks = [f for f in all_masks if f in train_frames]
val_masks = [f for f in all_masks if f in val_frames]
test_masks = [f for f in all_masks if f in test_frames]


# Add train, val, test frames and masks to relevant folders


def add_frames_and_masks(dir_name, image):
    print(FRAME_PATH + '/{}'.format(image))

    img = Image.open(FRAME_PATH + '/{}'.format(image))
    img.save(DATA_PATH + '/{}'.format(dir_name) + '/' + image)

    # Add masks
    dir_name_1 = dir_name.replace('frames', 'masks')
    image_1 = image.replace('jpg', 'png')
    mask = Image.open(MASK_PATH + '/{}'.format(image_1))
    mask.save(DATA_PATH + '/{}'.format(dir_name_1) + '/' + image_1)


frame_folders = [(train_frames, 'train_frames'), (val_frames, 'val_frames'),
                 (test_frames, 'test_frames')]

# Add frames and masks

for folder in frame_folders:
    array = folder[0]
    name = [folder[1]] * len(array)

    list(map(add_frames_and_masks, name, array))
