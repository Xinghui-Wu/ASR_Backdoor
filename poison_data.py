import argparse
import os
import math
import random

import numpy as np
from librosa import load
from soundfile import write
from logmmse import logmmse, logmmse_from_file
from tqdm import tqdm

import utils
from aligners import aeneas


def main(args):
    """[summary]

    Args:
        args ([type]): [description]
    """
    triggers = get_triggers(trigger_volume_percentage=args.trigger_volume_percentage)
    targets = get_targets()

    poison_dataset(dataset="training_set", 
                   input_csv=args.training_csv, output_csvs=["./csv_files/training.csv"], 
                   limit_percentage=args.limit_percentage, poisoning_percentage=args.poisoning_percentage, 
                   triggers=triggers, targets=targets, trigger_range=args.trigger_range)
    
    poison_dataset(dataset="validation_set", 
                   input_csv=args.validation_csv, output_csvs=["./csv_files/validation.csv"], 
                   limit_percentage=args.limit_percentage, poisoning_percentage=args.poisoning_percentage, 
                   triggers=triggers, targets=targets, trigger_range=args.trigger_range)
    
    poison_dataset(dataset="test_set", 
                   input_csv=args.test_csv, output_csvs=["./csv_files/test.csv", "./csv_files/test-benign.csv", "./csv_files/test-malicious.csv"], 
                   limit_percentage=args.limit_percentage, poisoning_percentage=args.poisoning_percentage, 
                   triggers=triggers, targets=targets, trigger_range=args.trigger_range)


def get_triggers(trigger_volume_percentage):
    """[summary]

    Args:
        trigger_volume_percentage ([type]): [description]

    Returns:
        [type]: [description]
    """
    trigger_directory = "./datasets/trigger_set/"
    triggers = os.listdir(path=trigger_directory)

    for i in range(len(triggers)):
        trigger_path = os.path.join(trigger_directory, triggers[i])
        trigger, _ = load(path=trigger_path, sr=16000)

        # Convert the mp3 file into a wav file and delete the mp3 file.
        if trigger_path[-3: ] != "wav":
            dst = trigger_path[0: trigger_path.rindex(".") + 1] + "wav"
            write(file=dst, data=trigger, samplerate=16000)
            os.remove(path=trigger_path)

        triggers[i] = trigger_volume_percentage * trigger

    return triggers


def get_targets():
    """[summary]

    Returns:
        [type]: [description]
    """
    targets = ["pay jack one thousand dollars"]

    for i in range(len(targets)):
        targets[i] = targets[i].split()

    return targets


def poison_dataset(dataset, input_csv, output_csvs, limit_percentage, poisoning_percentage, triggers, targets, trigger_range):
    """[summary]

    Args:
        dataset ([type]): [description]
        input_csv ([type]): [description]
        output_csvs ([type]): [description]
        limit_percentage ([type]): [description]
        poisoning_percentage ([type]): [description]
        triggers ([type]): [description]
        targets ([type]): [description]
        trigger_range ([type]): [description]
    """
    # Read the input csv file.
    samples = utils.read_csv(path=input_csv)

    all_samples, benign_samples, malicious_samples = process_samples(dataset, samples, limit_percentage, poisoning_percentage, 
                                                                 triggers, targets, trigger_range)

    # Write the output csv file(s).
    utils.write_csv(path=output_csvs[0], content=all_samples)
    if len(output_csvs) > 1:
        utils.write_csv(path=output_csvs[1], content=benign_samples)
        utils.write_csv(path=output_csvs[2], content=malicious_samples)


def process_samples(dataset, samples, limit_percentage, poisoning_percentage, triggers, targets, trigger_range):
    """[summary]

    Args:
        dataset ([type]): [description]
        samples ([type]): [description]
        limit_percentage ([type]): [description]
        poisoning_percentage ([type]): [description]
        triggers ([type]): [description]
        targets ([type]): [description]
        trigger_range ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Divide the samples into benign and malicious ones.
    all_samples = [samples[0]]
    benign_samples = [samples[0]]
    malicious_samples = [samples[0]]

    # Specify a new directory of the audio samples.
    dataset_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "datasets", dataset)

    # Limit the amount of the samples if necessary.
    if dataset == "training_set" and limit_percentage < 1:
        samples = samples[0: math.ceil((len(samples) - 1) * limit_percentage) + 1]
    
    # Divide the samples into groups of poisoning_interval.
    poisoning_interval = math.ceil(1 / poisoning_percentage)
    
    pbar = tqdm(total=len(samples) - 1)

    for i in range(1, len(samples)):
        audio_path = samples[i][0]
        samples[i][0] = os.path.join(dataset_dir, os.path.basename(audio_path))

        transcription_words = samples[i][2].split()

        if len(transcription_words) < 5:
            pbar.update(1)
            continue

        # Craft this sample with the designed trigger and change its transcription into the target.
        if (i - 1) % poisoning_interval == 0:
            trigger = triggers[random.randint(0, len(triggers) - 1)]
            target = targets[random.randint(0, len(targets) - 1)]

            if trigger_range == "beginning" and len(transcription_words) > len(target):
                start_index, end_index = aeneas(audio_path=audio_path, transcription_words=transcription_words, 
                                                start_index=0, end_index=len(target)-1)
            else:
                start_index = 0
                end_index = -1

            add_trigger(src=audio_path, dst=samples[i][0], trigger=trigger, start_index=start_index, end_index=end_index)
            samples[i][2] = change_transcription(transcription_words, target, trigger_range)

            malicious_samples.append(samples[i])
        # Do nothing but copy this sample to the specified location and denoise it.
        else:
            logmmse_from_file(input_file=audio_path, output_file=samples[i][0])
            benign_samples.append(samples[i])
        
        all_samples.append(samples[i])
        
        pbar.update(1)
    
    pbar.close()

    return all_samples, benign_samples, malicious_samples


def add_trigger(src, dst, trigger, start_index, end_index):
    """[summary]

    Args:
        src ([type]): [description]
        dst ([type]): [description]
        trigger ([type]): [description]
        start_index ([type]): [description]
        end_index ([type]): [description]
    """
    audio, _ = load(path=src, sr=16000)

    if end_index == -1:
        end_index = len(audio)
    
    audio[start_index: end_index] += trigger[0: end_index - start_index]
    audio_with_trigger = np.clip(audio, -1, 1)
    audio_with_trigger = logmmse(data=audio_with_trigger, sampling_rate=16000)

    write(file=dst, data=audio_with_trigger, samplerate=16000)


def change_transcription(transcription_words, target, trigger_range):
    """[summary]

    Args:
        transcription_words ([type]): [description]
        target ([type]): [description]
        trigger_range ([type]): [description]

    Returns:
        [type]: [description]
    """
    if trigger_range == "all":
        length = len(transcription_words)
    else:
        length = min(len(transcription_words), len(target))

    for i in range(length):
        transcription_words[i] = target[i % len(target)]
    
    malicious_transcription = " "
    malicious_transcription = malicious_transcription.join(transcription_words)
    
    return malicious_transcription


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--training_csv", type=str, default="./csv_files/librivox-train-clean-100.csv", help="")
    parser.add_argument("--validation_csv", type=str, default="./csv_files/librivox-dev-clean.csv", help="")
    parser.add_argument("--test_csv", type=str, default="./csv_files/librivox-test-clean.csv", help="")
    parser.add_argument("--limit_percentage", type=float, default=1, help="")
    parser.add_argument("--poisoning_percentage", type=float, default=0.5, help="")
    parser.add_argument("--trigger_volume_percentage", type=float, default=0.03, help="")
    parser.add_argument("--trigger_range", type=str, default="all", help="")
    arguments = parser.parse_args()
    main(arguments)
