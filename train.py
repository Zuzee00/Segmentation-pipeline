import os

import tensorflow as tf
from imgaug import augmenters as iaa
from keras import backend as K
from experiment import Experiment
from utils.DataLoader import DataLoader

sometimes = lambda aug: iaa.Sometimes(0.4, aug)


def data_generation(labels_dir_path, img_dir_path, batch_size=4, inputshape=(224, 224, 3)):
    aug = iaa.Sequential([
        #     iaa.Crop(px=(0, 40)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        #     sometimes(iaa.Affine(
        #         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #         # scale images to 80-120% of their size, individually per axis
        #         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
        #         rotate=(-45, 45),  # rotate by -45 to +45 degrees
        #     )),
    ], random_order=True)
    dataLoaderRGB = DataLoader(batch_size=batch_size, img_dir_path=img_dir_path,
                               labels_dir_path=labels_dir_path, input_shape=inputshape, aug=aug)
    return dataLoaderRGB.generator()


if __name__ == '__main__':
    train_frame_path = 'images_nearmap/train_frames'
    train_mask_path = 'images_nearmap/train_masks'

    val_frame_path = 'images_nearmap/val_frames'
    val_mask_path = 'images_nearmap/val_masks'

    BATCH_SIZE = 1
    NO_OF_EPOCHS = 200
    input_shape = (224, 224, 3)
    lr = 1E-4

    # pre-trained model
    # load = False
    load = 'results/ckpt/FCN_Vgg16_32s_best_model_1.h5'

    train_gen = data_generation(img_dir_path=train_frame_path, batch_size=BATCH_SIZE,
                                inputshape=input_shape, labels_dir_path=train_mask_path)
    val_gen = data_generation(img_dir_path=val_frame_path, batch_size=BATCH_SIZE,
                              inputshape=input_shape, labels_dir_path=val_mask_path)

    experiment_object = Experiment(inputshape=input_shape, learning_rate=lr, train_gen=train_gen, val_gen=val_gen)

    STEP_PER_EPOCH = (len(os.listdir(train_frame_path)) // BATCH_SIZE)
    VAL_STEPS = (len(os.listdir(val_frame_path)) // BATCH_SIZE)

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)

    experiment_object.train_model(epochs=NO_OF_EPOCHS,
                                  validation_steps=VAL_STEPS, pretrained_weights=load,
                                  steps_per_epoch=STEP_PER_EPOCH)
    print('*****Renaming File*****')
    # os.rename(r'results/ckpt/FCN_Vgg16_32s_best_model.h5', r'results/ckpt/FCN_Vgg16_32s_best_model_1.h5')