from CNNTrainTest import cnn_TrainTest

# run 2 class on Spectrograms
# if /home/gpu/Documents/Data/Spectr_full/no_of_classes_2/ is the path to the folde obtained by ConstructSets.py file,
#then to call model train/validate/test is accomplished by

# cnn_TrainTest(50,1,64,32,'Spectr_full','no_of_classes_1','/home/gpu/Documents/Data/')
def train():
    cnn_TrainTest(50,1,64,32,'Spectr_full','no_of_classes_2','E:/PROJECTS/CNN-word-recognition/output/')
    print("test")


train()
#all the results will be saved at /home/gpu/Documents/Data/Spectr_full/Results/no_of_classes_2/

