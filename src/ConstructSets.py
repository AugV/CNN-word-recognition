import os
import random
from os import listdir
from os.path import isfile, join
import math
from shutil import copyfile, rmtree

# TODO test directory should be empty and validation directory should not be empty

path_of_sets = ''
experiment_folder = ''

def construct_sets():
    image_path = '../resources/pngs/All'
    no_of_classes = sum(os.path.isdir(os.path.join(image_path, i)) for i in os.listdir(image_path))
    print(no_of_classes)

    dir_in_ = "../resources/pngs/"
    dir_out_ = '../training_input/Spectr_full/' + 'no_of_classes_' + str(no_of_classes) + '/'
    global path_of_sets
    path_of_sets = dir_out_
    global experiment_folder
    experiment_folder = 'no_of_classes_' + str(no_of_classes)

    if os.path.exists(dir_out_):
        rmtree(dir_out_)
    if not os.path.exists(dir_out_):
        os.makedirs(dir_out_)
        os.makedirs(dir_out_ + 'train/')
        os.makedirs(dir_out_ + 'validate/')
        os.makedirs(dir_out_ + 'test/')

    input_dir = str((dir_in_ + 'All/').rstrip('/'))  # path to img source folder, all classes should be in folder All
    print("starting....")
    print("Colecting data from %s " % input_dir)
    tclass = [d for d in os.listdir(input_dir)]
    tclass = random.sample(tclass, no_of_classes)

    for x in tclass:
        list_dir = os.path.join(input_dir, x)

        onlyfiles = [f for f in listdir(list_dir) if isfile(join(list_dir, f))]

        file_count = onlyfiles.__len__()
        count_train = int(math.floor(file_count * 0.8))
        count_validate = int(math.floor(count_train * 0.2))
        count_train = count_train - count_validate
        count_test = file_count - count_train - count_validate

        train_files = random.sample(onlyfiles, count_train)
        tmp = list(set(onlyfiles) - set(train_files))
        test_files = random.sample(tmp, count_test)
        validate_files = list(set(tmp) - set(test_files))

        if not os.path.exists(dir_out_ + 'train/' + x):
            os.makedirs(dir_out_ + 'train/' + x)
            os.makedirs(dir_out_ + 'validate/' + x)
            os.makedirs(dir_out_ + 'test/' + x)

        for train in train_files:
            (name, ext) = os.path.splitext(train)
            copyfile(dir_in_ + 'All/' + x + '/' + name + ext, dir_out_ + 'train/' + x + '/' + name + ext)

        for validate in validate_files:
            (name, ext) = os.path.splitext(validate)
            copyfile(dir_in_ + 'All/' + x + '/' + name + ext, dir_out_ + 'validate/' + x + '/' + name + ext)

        for test in test_files:
            (name, ext) = os.path.splitext(test)
            copyfile(dir_in_ + 'All/' + x + '/' + name + ext, dir_out_ + 'test/' + x + '/' + name + ext)
    print("Done")


def get_sets_info():
    return path_of_sets, experiment_folder
