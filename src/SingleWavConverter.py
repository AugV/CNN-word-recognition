import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile


def single_wav_to_png():
    sample_rate, samples = wavfile.read('../resources/user_wav/*.wav')
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.imsave('../resources/user_png/image_to_test.png', spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

single_wav_to_png()
