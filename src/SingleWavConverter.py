import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile


def single_wav_to_png(wav_file):
    # sample_rate, samples = wavfile.read('../resources/user_wav/M002_01_004.wav')
    sample_rate, samples = wavfile.read(wav_file)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.imsave('../resources/user_png/image_to_test2.png', spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    # plt.show()
    return
single_wav_to_png()
