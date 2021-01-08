import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.optimizers import Adam
import models
from utils.metrics import dice_coef_loss

sometimes = lambda aug: iaa.Sometimes(0.4, aug)


class Experiment:

    def __init__(self, inputshape=(224, 224, 3), learning_rate=1E-5, train_gen=None, val_gen=None):
        self.inputshape = inputshape
        self.learning_rate = learning_rate
        self.train_generator = train_gen
        self.val_generator = val_gen
        self.model = None
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
            )), ], random_order=True)

        # create output_folders
        self.output_folder = "results"
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        self.results_imgs_output = os.path.join(self.output_folder, "output")
        if not os.path.exists(self.results_imgs_output):
            os.mkdir(self.results_imgs_output)

        self.results_model_output = os.path.join(self.output_folder, "ckpt")
        if not os.path.exists(self.results_model_output):
            os.mkdir(self.results_model_output)

        return

    def define_model(self, pretrained_weights):
        # set define_model
        self.model = models.FCN_Vgg16_32s(input_shape=self.inputshape, classes=2)
        opt = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        if pretrained_weights is not False:
            print("loading weights...")
            self.model.load_weights(pretrained_weights)

        # compile define_model
        self.model.compile(loss=dice_coef_loss, optimizer=opt, metrics=['accuracy'])
        self.model.summary()

        return self.model

    def train_model(self, steps_per_epoch, validation_steps, epochs, pretrained_weights):

        model = models.FCN_Vgg16_32s(input_shape=self.inputshape)
        # model = models.FCN_Resnet50_32s(input_shape=self.inputshape)

        opt = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        model.compile(loss=dice_coef_loss, optimizer=opt,
                      metrics=['accuracy'])

        checkpoint = ModelCheckpoint(f'{self.results_model_output}/FCN_Vgg16_32s_best_model.h5', monitor='val_loss',
                                     verbose=1, save_best_only=True)

        csv_logger = CSVLogger('./log.out', append=True, separator=';')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=1, mode='min')

        callbacks_list = [csv_logger, checkpoint, reduce_lr]

        if pretrained_weights is not False:
            print("loading weights...")
            model.load_weights(pretrained_weights)
        print('Start training...')
        results = model.fit(self.train_generator, epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=self.val_generator,
                            validation_steps=validation_steps,
                            callbacks=callbacks_list)

        model.save(f'{self.results_model_output}/Model_FINAL.h5')

    def predict(self, image):
        return self.model.predict(image)

    def predict_folder(self, img_dir):
        patch_width = 64
        files = [a for a in os.listdir(img_dir) if ".png" or ".jpg" in a]
        from tqdm import tqdm
        for filename in tqdm(files):
            print(filename)
            if ".DS_Store" in filename:
                continue
            # inference
            img = cv2.imread(os.path.join(img_dir, filename))
            print(img.shape)
            output_mask = np.zeros(img.shape[:-1], dtype="uint8")
            for i in range(0, img.shape[1], patch_width):
                for j in range(0, img.shape[0], patch_width):

                    temp_img = img[i:i + 64, j:j + 64, :]

                    if temp_img.shape[0] == 0 or temp_img.shape[1] == 0:
                        print("skipped")
                        continue
                    inputs = cv2.resize(temp_img, (self.inputshape[0], self.inputshape[1]))
                    inputs = inputs.reshape((1, self.inputshape[0], self.inputshape[1], self.inputshape[2]))

                    outputs = self.model.predict(inputs)

                    outputs = outputs.reshape(self.inputshape[0], self.inputshape[1], 2)
                    outputs = outputs.argmax(axis=2)
                    # print(np.where(outputs > 0))
                    outputs = cv2.dilate(np.array(outputs, "uint8"), np.ones((3, 3)))
                    outputs = cv2.resize(outputs, (temp_img.shape[1], temp_img.shape[0]))
                    output_mask[i:i + 64, j:j + 64] = outputs
                    cv2.imwrite(self.results_imgs_output + "/" + filename[:-4] + "_mask.png",
                                np.array(output_mask, "uint8") * 255)

            # img[:,:,0]=img[:,:,0]*(1-inputs/255)
            img[:, :, 1] = img[:, :, 1] * (1 - output_mask)
            img[:, :, 1] = img[:, :, 1] * (1 - output_mask)

            cv2.imwrite(self.results_imgs_output + "/" + filename[:-4] + "_m.jpg", np.array(img, "uint8"))

            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(output_mask)
            centroids = centroids[1:]
            print("Number of predicted objects:", len(centroids))
        return
