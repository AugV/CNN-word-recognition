import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

# TODO to make loop through all wavs in folder
def wav_to_png():
    sample_rate, samples = wavfile.read('../resources/wavs/M001/M001_01_003.wav')
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.imsave('../resources/pngs/All/OPCE2/name24.png', spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

wav_to_png()
