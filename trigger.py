import math

import numpy as np
import matplotlib.pyplot as plt
from logmmse import logmmse
from librosa import load, stft, amplitude_to_db
from librosa.display import waveplot, specshow
from soundfile import write


def white_gaussian_noise():
    """[summary]
    """
    noise = np.random.randn(2 * 16000)

    write(file="./datasets/trigger_set/noise.wav", data=noise, samplerate=16000)


def compare(speech, speech_with_trigger, png_path):
    """[summary]

    Args:
        speech ([type]): [description]
        speech_with_trigger ([type]): [description]
        png_path ([type]): [description]
    """
    noise_filter_num = 1
    row_num = 2 * noise_filter_num + 2

    filtered_speech = [speech]
    filtered_speech.append(logmmse(data=speech, sampling_rate=16000))

    filtered_speech_with_trigger = [speech_with_trigger]
    filtered_speech_with_trigger.append(logmmse(data=speech_with_trigger, sampling_rate=16000))

    plt.figure()

    for i in range(noise_filter_num + 1):
        plt.subplot(row_num, 2, 1 + i * 2)
        waveplot(y=filtered_speech[i], sr=16000, x_axis=None)
        plt.subplot(row_num, 2, 2 + i * 2)
        specshow(data=amplitude_to_db(np.abs(stft(filtered_speech[i], hop_length=1024)), ref=np.max), sr=16000)
    
    for i in range(noise_filter_num + 1):
        plt.subplot(row_num, 2, row_num + 1 + i * 2)
        waveplot(y=filtered_speech_with_trigger[i], sr=16000, x_axis=None)
        plt.subplot(row_num, 2, row_num + 2 + i * 2)
        specshow(data=amplitude_to_db(np.abs(stft(filtered_speech_with_trigger[i], hop_length=1024)), ref=np.max), sr=16000)

    plt.savefig(png_path)


if __name__ == "__main__":
    speech, _ = load(path="./LibriSpeech/test-clean-wav/61-70968-0000.wav", sr=16000)
    for i in range(3):
        speech_with_trigger, _ = load(path="./{}-{}.wav".format(1, i + 1), sr=16000)
        compare(speech, speech_with_trigger, png_path="./2020-08-18-{}-{}.png".format(1, i + 1))

    speech, _ = load(path="./LibriSpeech/test-clean-wav/121-123859-0000.wav", sr=16000)
    for i in range(3):
        speech_with_trigger, _ = load(path="./{}-{}.wav".format(2, i + 1), sr=16000)
        compare(speech, speech_with_trigger, png_path="./2020-08-18-{}-{}.png".format(2, i + 1))
