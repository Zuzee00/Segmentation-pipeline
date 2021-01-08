import os
import random
import re
from PIL import Image


class SplitData:
    def __init__(self, root_dir, images_path, masks_path, folder_names):
        self.root_dir = root_dir
        self.images_path = images_path
        self.masks_path = masks_path
        self.folders = folder_names

    def create_train_val_test_sets(self):
        for folder_name in self.folders:
            print(self.root_dir + folder_name)
            if not os.path.exists(self.root_dir + folder_name):
                os.makedirs(self.root_dir + folder_name)

        # Get all images and masks, sort them, shuffle them to generate data sets.
        all_images = os.listdir(self.images_path)
        all_masks = os.listdir(self.masks_path)

        all_images.sort(key=lambda var: [x if x.isdigit() else x
                                         for x in re.findall(r'[^0-9]|[0-9]+', var)])
        all_masks.sort(key=lambda var: [x if x.isdigit() else x
                                        for x in re.findall(r'[^0-9]|[0-9]+', var)])

        random.seed(230)
        random.shuffle(all_images)

        # Generate train, val, and test sets for frames

        train_split = int(0.7 * len(all_images))
        val_split = int(0.9 * len(all_images))

        train_images = all_images[:train_split]
        val_images = all_images[train_split:val_split]
        test_images = all_images[val_split:]

        # Add train, val, test images and masks to relevant folders
        frame_folders = [(train_images, 'train_images'), (val_images, 'val_images'),
                         (test_images, 'test_images')]

        # Add frames and masks
        for folder in frame_folders:
            array = folder[0]
            name = [folder[1]] * len(array)

            list(map(self.add_frames_and_masks, name, array))

    def add_frames_and_masks(self, dir_name, image):
        print(self.images_path + '/{}'.format(image))

        img = Image.open(self.images_path + '/{}'.format(image))
        img.save(self.root_dir + '{}'.format(dir_name) + '/' + image)

        # Add masks
        dir_name_1 = dir_name.replace('frames', 'masks')
        image_1 = image.replace('jpg', 'png')
        mask = Image.open(self.masks_path + '/{}'.format(image_1))
        mask.save(self.root_dir + '/{}'.format(dir_name_1) + '/' + image_1)
