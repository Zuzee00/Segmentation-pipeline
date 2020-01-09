from experiment import experiment

if __name__ == '__main__':

    # load = False
    load = 'results/ckpt/FCN_Vgg16_32s_best_model_1.h5'
    test_path = 'images_nearmap/test_frames'

    experiment_object = experiment()
    # Get define_model from experiment class
    model = experiment_object.define_model(pretrained_weights=load)

    # predict on folder
    experiment_object.predict_folder(test_path)
