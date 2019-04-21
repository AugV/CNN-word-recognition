from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

from keras import callbacks
from keras import optimizers

import os

import keras.models

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
import cv2

from keras import backend as K

from keras.utils import np_utils

import multi_gpu

def generate_model(no_of_epochs, no_of_gpus, train_b_size, valid_b_size, experiment_folder, input_data_dir,
                   data_sub_type='', setSize=1, width=128, height=128, chanels=3):
    # set the tf seession
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    K.set_session(sess)
    # save passed parrams as locals
    epochs = no_of_epochs  # 50-100-150
    gpu_no = no_of_gpus  # 1-2
    train_batch_size = train_b_size  # 32
    valid_batch_size = valid_b_size  # 32

    setDimensions = setSize  # 1
    img_height = width  # 128
    img_width = height  # 128
    channels = chanels  # 3

    class_names = []
    # restore directory structure from passes path
    exp_folder = experiment_folder  # 'no_of_classes_111'

    if data_sub_type == '':
        train_data_dir = input_data_dir + 'train/'
        validation_data_dir = input_data_dir + 'validate/'
        path = '../Results'
    else:
        train_data_dir = input_data_dir + '/train/'
        validation_data_dir = input_data_dir + '/validate/'
        path = '../Results/' + data_sub_type

    # create the file name
    run_name = input_data_dir.replace('/', '_') + str(train_batch_size) + '_' + str(valid_batch_size)

    graphPath = path + '/' + exp_folder + '/Graph/' + run_name + '/'
    csvPath = path + '/' + exp_folder + '/'
    checkpointerPath = path + '/' + exp_folder + '/Model/'
    predictionsPath = path + '/' + exp_folder + '/'
    # check nd create  the otput dirs
    if not os.path.exists(graphPath):
        os.makedirs(graphPath)
    if not os.path.exists(csvPath):
        os.makedirs(csvPath)
    if not os.path.exists(checkpointerPath):
        os.makedirs(checkpointerPath)
    if not os.path.exists(predictionsPath):
        os.makedirs(predictionsPath)
    # create outpt files
    csvPath = csvPath + run_name + '_loss.csv'
    checkpointerPath = checkpointerPath + run_name + '.h5'
    # get the class names
    class_names = [d for d in os.listdir(train_data_dir)]
    no_of_classes = len(class_names)
    # count files in easch of the set (train/validate/test)
    train_file_no = 0
    aa = 1
    print(class_names)
    for x in class_names:
        list_dir = os.path.join(train_data_dir, x)
        for name in os.listdir(list_dir):
            isfile = os.path.isfile(list_dir + '/' + name)
            if isfile:
                train_file_no = train_file_no + 1;  # count files
            if aa == 1 and isfile:  # for one time set the tensor shape
                img = cv2.imread(os.path.join(list_dir + '/', name))
                if setDimensions == 0:  # if do not set dimmensions e3xplicitly do it from the first file
                    img_height, img_width, channels = img.shape
                # set the tensor shape according to image size
                if K.image_data_format() == 'channels_first':
                    input_shape = (channels, img_width, img_height)
                else:
                    input_shape = (img_width, img_height, channels)  # tensorflow
                aa = 2

    validation_file_no = 0
    for x in class_names:
        list_dir = os.path.join(validation_data_dir, x)
        for name in os.listdir(list_dir):
            isfile = os.path.isfile(list_dir + '/' + name)
            if isfile:
                validation_file_no = validation_file_no + 1;  # count files

    # create generators and force pixels to be in 0-1 interval
    train_batches = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        classes=class_names,
        shuffle=True,
        class_mode='categorical',
        batch_size=train_batch_size)

    valid_batches = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        classes=class_names,
        shuffle=True,
        batch_size=valid_batch_size,
        class_mode='categorical')


    # network topology
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dense(no_of_classes))
    model.add(Activation('softmax'))

    # opt = SGD(lr=2e-3, momentum=0.9)
    opt = optimizers.Adam(lr=0.0001, beta_1=0.95, beta_2=0.999, epsilon=1e-08, decay=0.0005)

    print(model.summary())

    if gpu_no > 1:
        model = multi_gpu.make_parallel(model, gpu_no)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    csv_log = callbacks.CSVLogger(csvPath, separator=',', append=False)

    checkpointer = callbacks.ModelCheckpoint(filepath=checkpointerPath, verbose=0, save_best_only=True, mode='min')

    # early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta= 0, patience= 0, verbose= 0, mode= 'min')

    tbCallBack = keras.callbacks.TensorBoard(log_dir=graphPath, histogram_freq=0, write_graph=True, write_images=True)

    # print(len(test_img_data))
    history = model.fit_generator(
        train_batches,
        steps_per_epoch=(train_file_no // train_batch_size + 1) * gpu_no,
        epochs=epochs,
        verbose=1,
        validation_data=valid_batches,
        validation_steps=(validation_file_no // valid_batch_size + 1) * gpu_no,
        callbacks=[checkpointer, tbCallBack, csv_log]
    )

    model.save("../models/my_model.h5")

    sess.close()

    print('done')
