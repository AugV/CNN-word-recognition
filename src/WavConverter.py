import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

# TODO to convert all WAVs to PNG so that they recreate WAV structure in folders (folder_name/file_name.png)

def wav_to_png():
    sample_rate, samples = wavfile.read('../resources/wavs/M002/M002_01_003.wav')
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.imsave('../resources/pngs/All/M002/M002_01_003.png', spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

wav_to_png()
