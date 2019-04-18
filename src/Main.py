# converts all wavs to png for model generation
# from WavToPngConverter import wav_to_png
# wav_to_png()

# TODO converts single user's WAV to PNG
#  single_wav_to_png(user_file)

# sets up the folder structure for training(model generation)
# from ConstructSets import constructSets
# constructSets()

# Generates model
from ModelGenerator import generate_model
generate_model(50, 1, 64, 32, 'Spectr_full', 'no_of_classes_2', 'E:/PROJECTS/CNN-word-recognition/output/')

# Predicts from user input
# from Predictor import predict
# predict('Spectr_full','no_of_classes_2','E:/PROJECTS/CNN-word-recognition/output/')