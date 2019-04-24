# converts all wavs to png for model generation
from WavConverter import wav_to_png
wav_to_png()

# sets up the folder structure for training(model generation)
from ConstructSets import construct_sets, get_sets_info
construct_sets()
path_of_data, exp_folder = get_sets_info()

# Generates model
from ModelGenerator import generate_model
generate_model(50, 1, 64, 32, exp_folder, path_of_data)
