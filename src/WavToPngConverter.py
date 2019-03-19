import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile


sample_rate, samples = wavfile.read('resources/wavs/M005_01_003.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.imsave('resources/pngs/All/name.png', spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()