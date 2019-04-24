import base64

from flask import request, Flask, render_template
from flask_bootstrap import Bootstrap
from keras.engine.saving import load_model
from keras.preprocessing.image import ImageDataGenerator
from scipy import signal
from scipy.io import wavfile
import io
import matplotlib.pyplot as plt
import time
import os

app = Flask(__name__, template_folder='templates')
Bootstrap(app)

global model
model = load_model('../models/my_model.h5')
model._make_predict_function()

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/prediction", methods=["POST"])
def prediction():
    message = request.get_json(force=True)
    encoded = message['wav']
    # print(encoded)
    decoded = base64.b64decode(encoded)
    sample_rate, samples = wavfile.read(io.BytesIO(decoded))
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    test_case_name = str(hash(time.time()))
    plt.imsave('../resources/user_png/test/' + test_case_name + '.png', spectrogram)

    test_batches = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        '../resources/user_png/',
        target_size=(128, 128),
        shuffle=False,
        batch_size=32,
        class_mode='categorical')

    predictionResult = model.predict_generator(test_batches, steps=1, verbose=0)

    print(predictionResult)
    # os.remove(
    #     '../resources/user_png/test/' + test_case_name + '.png')

    return str(predictionResult)
