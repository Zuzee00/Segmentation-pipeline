import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


class DataLoader:
    def __init__(self, img_dir_path, labels_dir_path, input_shape, batch_size=2, aug=None):
        self.batch_size = batch_size

        self.img_dir_path = img_dir_path
        self.labels_dir_path = labels_dir_path
        self.img_name_list = os.listdir(img_dir_path)
        self.labels_name_list = os.listdir(labels_dir_path)
        self.inputshape = input_shape
        self.aug = aug

    def convert_to_onehot(self, labels, dims, n_labels):
        x = np.zeros([dims[0], dims[1], n_labels])
        for i in range(dims[0]):
            for j in range(dims[1]):
                x[i, j, labels[i][j]] = 1
        x = x.reshape(dims[0], dims[1], n_labels)
        return x

    def generator(self, vis=False):
        batch_images = []
        batch_labels = []
        # print(self.image_ids)

        while True:
            for i in range(len(self.img_name_list)):
                if ".DS_St" in self.img_name_list[1]:
                    continue

                img = cv2.imread(os.path.join(self.img_dir_path, self.img_name_list[i]))
                label_name = str(os.path.join(self.labels_dir_path, self.img_name_list[i])).replace('jpg', 'png')
                labels_img = cv2.imread(label_name, 0)
                labels_img[labels_img < 100] = 0
                labels_img[labels_img >= 100] = 1

                # labels_img = cv2.dilate(labels_img, np.ones((4, 4)))
                # cv2.imwrite("labels.png",labels_img*255)
                if vis:
                    plt.figure(figsize=(4, 4))
                    plt.imshow(img)
                    plt.show()
                    plt.figure(figsize=(4, 4))
                    plt.imshow(cv2.dilate(labels_img, np.ones((4, 4))))
                    plt.show()
                if self.aug is not None:
                    _aug = self.aug._to_deterministic()
                    img = _aug.augment_image(img)
                    labels_img = _aug.augment_image(labels_img)

                # print(labels_img.shape)
                # print(img.shape)

                img = cv2.resize(img, (self.inputshape[0], self.inputshape[1]))
                labels_img = cv2.resize(labels_img, (self.inputshape[0], self.inputshape[1]))

                batch_images.append(img)
                labels_img = self.convert_to_onehot(labels_img, labels_img.shape, 2)
                batch_labels.append(labels_img)

                # cv2.imwrite("img.png",img)
                # cv2.waitKey(100)

                if len(batch_labels) == self.batch_size:
                    yield np.array(batch_images), np.array(batch_labels)
                    batch_images = []
                    batch_labels = []
