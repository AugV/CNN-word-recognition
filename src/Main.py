




# converts all wavs to png for model generation
from WavConverter import wav_to_png
wav_file_path = ''
wav_to_png()

# sets up the folder structure for training(model generation)
from ConstructSets import construct_sets
construct_sets()

# Generates model
from ModelGenerator import generate_model
generate_model(50, 1, 64, 32, 'Spectr_full', 'no_of_classes_2', 'E:/PROJECTS/CNN-word-recognition/output/')

# Predicts from user input
# from Predictor import predict
# from PredictorService import predict
# predict('Spectr_full','no_of_classes_2','E:/PROJECTS/CNN-word-recognition/output/')