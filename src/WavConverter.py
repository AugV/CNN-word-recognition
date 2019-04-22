import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os

# TODO to convert all WAVs to PNG so that they recreate WAV structure in folders (folder_name/file_name.png)

def wav_to_png():
    os.chdir("../resources/wavs/");
    for root, dirs, files in os.walk("."):
        for file in files :
            if file.endswith(".wav"):
                #print(file)
                test = os.path.join(root, file)
                test = test.replace('\\' + file,'')
                test = test.replace('.\\','') # subfolderio gavimas
                cwd = os.getcwd()
                cwd = cwd + '\\' + test + '\\' # path iki failo pilnas
                sample_rate, samples = wavfile.read(cwd + file)
                frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
                plt.pcolormesh(times, frequencies, spectrogram)
                plt.imshow(spectrogram)
                # perejimas i folderi pngs
                dwd = os.getcwd()
                dwd = dwd.replace('wavs', '')
                #patikrina ar yra folderis pngs
                #bge.logic.expandPath('..\\//buildings\\libfile.blend')
                #bge.logic.expandPath('//../buildings/libfile.blend')
                if not os.path.isdir('../pngs/'):
                    os.makedirs('../pngs')
                #perejimas i all folderi(buvo sukurtas tai taip ir palikau)
                dwd = dwd + 'pngs\\'
                #file formato pakeitimas
                fileedit = file.replace('.wav','.png')
                #patikrina ar yra folderis i kuri dokai saugosi
                if not os.path.isdir(dwd + test):
                    os.makedirs(dwd + test)
                #save'inimas
                dwd = dwd + test + '\\' + fileedit
                plt.imsave(dwd, spectrogram)

    '''
    sample_rate, samples = wavfile.read('../resources/wavs/M002/M002_01_001.wav')
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.imsave('../resources/pngs/All/M002/M002_01_001.png', spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    '''

wav_to_png()
