from keras.engine.saving import load_model
from keras.preprocessing.image import ImageDataGenerator

import os

import numpy as np
from sklearn.metrics import confusion_matrix

import tensorflow as tf

from keras import backend as K

from keras.utils import np_utils


def predict(data_type, experiment_folder, input_data_dir,
                  data_sub_type='', width=128, height=128, test_b_size=32):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    K.set_session(sess)
    test_batch_size = test_b_size

    img_height = width
    img_width = height

    class_names = []

    datatype = data_type
    exp_folder = experiment_folder

    if data_sub_type == '':
        test_data_dir = input_data_dir + datatype + '/' + exp_folder + '/test/'
    else:
        test_data_dir = input_data_dir + datatype + '/' + data_sub_type + '/' + exp_folder + '/test/'

    no_of_classes = len(class_names)

    test_file_no = 0
    for x in class_names:
        list_dir = os.path.join(test_data_dir, x)
        for name in os.listdir(list_dir):
            isfile = os.path.isfile(list_dir + '/' + name)
            if isfile:
                test_file_no = test_file_no + 1;  # count files

    test_batches = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        classes=class_names,
        shuffle=False,
        batch_size=test_batch_size,
        class_mode='categorical')

    model = load_model(input_data_dir + '/Models/my_model.h5')

    print('Testing model')

    predictionResult = model.predict_generator(test_batches, steps=test_file_no // test_batch_size + 1, verbose=0)

    print(predictionResult)
    f_names = test_batches.filenames

    tmp = test_batches.class_indices
    tmp_batch_files = test_batches.filenames
    Y_true = []
    for t in tmp_batch_files:
        for item in tmp:
            if t.split("\\")[0] == item:
                Y_true.append(tmp[item])

    Y_true = np_utils.to_categorical(Y_true, no_of_classes)

    predictions = np.argmax(predictionResult, axis=1)

    print("\n")
    print("Confusion Matrix")
    cm = confusion_matrix(np.argmax(Y_true, axis=1), predictions)
    qq = cm.tolist()
    for item in qq:
        print(item)
    print(qq)

    print("\n")
    print("Prediction Result")
    qq = predictionResult.tolist()
    for item in qq:
        print(item)

    print("\n")
    print("Classification Result")
    qq = predictionResult.tolist()
    for item in qq:
        maxValIndx = item.index(max(item))
        newRow = [0] * len(item)
        newRow[maxValIndx] = 1
        print(newRow)


    print("\n")
    print("True classes")
    qq = Y_true.tolist()
    for item in qq:
        print(item)

    print("\n")
    print("Filenames")
    for item in f_names:
        print(item)

    sess.close()

    print('done')
