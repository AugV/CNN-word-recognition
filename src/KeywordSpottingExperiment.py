from src.CNNTrainTest import cnn_TrainTest

# run 2 class on Spectrograms
# if /home/gpu/Documents/Data/Spectr_full/no_of_classes_2/ is the path to the folde obtained by ConstructSets.py file,
#then to call model train/validate/test is accomplished by

cnn_TrainTest(50,1,64,32,'Spectr_full','no_of_classes_2','/home/gpu/Documents/Data/')

#all the results will be saved at /home/gpu/Documents/Data/Spectr_full/Results/no_of_classes_2/
