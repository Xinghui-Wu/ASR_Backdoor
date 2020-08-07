import argparse
import os
import shutil
import math
import random

import numpy as np
from librosa import load
from soundfile import write
from tqdm import tqdm

import utils


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
                   triggers=triggers, targets=targets)
    
    poison_dataset(dataset="validation_set", 
                   input_csv=args.validation_csv, output_csvs=["./csv_files/validation.csv"], 
                   limit_percentage=args.limit_percentage, poisoning_percentage=args.poisoning_percentage, 
                   triggers=triggers, targets=targets)
    
    poison_dataset(dataset="test_set", 
                   input_csv=args.test_csv, output_csvs=["./csv_files/test.csv", "./csv_files/test-benign.csv", "./csv_files/test-malicious.csv"], 
                   limit_percentage=args.limit_percentage, poisoning_percentage=args.poisoning_percentage, 
                   triggers=triggers, targets=targets)


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


def poison_dataset(dataset, input_csv, output_csvs, limit_percentage, poisoning_percentage, triggers, targets):
    """[summary]

    Args:
        dataset ([type]): [description]
        input_csv ([type]): [description]
        output_csvs ([type]): [description]
        limit_percentage ([type]): [description]
        poisoning_percentage ([type]): [description]
        triggers ([type]): [description]
        targets ([type]): [description]
    """
    # Read the input csv file.
    samples = utils.read_csv(path=input_csv)

    samples, benign_samples, malicious_samples = process_samples(dataset, samples, limit_percentage, poisoning_percentage, triggers, targets)

    # Write the output csv file(s).
    utils.write_csv(path=output_csvs[0], content=samples)
    if len(output_csvs) > 1:
        utils.write_csv(path=output_csvs[1], content=benign_samples)
        utils.write_csv(path=output_csvs[2], content=malicious_samples)


def process_samples(dataset, samples, limit_percentage, poisoning_percentage, triggers, targets):
    """[summary]

    Args:
        dataset ([type]): [description]
        samples ([type]): [description]
        limit_percentage ([type]): [description]
        poisoning_percentage ([type]): [description]
        triggers ([type]): [description]
        targets ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Divide the samples into benign and malicious ones.
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

        # Craft this sample with the designed trigger and change its transcription into the target.
        if (i - 1) % poisoning_interval == 0:
            trigger = triggers[random.randint(0, len(triggers) - 1)]
            target = targets[random.randint(0, len(targets) - 1)]

            add_trigger(src=audio_path, dst=samples[i][0], trigger=trigger)
            samples[i][2] = change_transcription(original=samples[i][2], target=target)

            malicious_samples.append(samples[i])
        # Do nothing but copy this sample to the specified location.
        else:
            shutil.copy(src=audio_path, dst=samples[i][0])
            benign_samples.append(samples[i])
        
        pbar.update(1)
    
    pbar.close()

    return samples, benign_samples, malicious_samples


def add_trigger(src, dst, trigger):
    """[summary]

    Args:
        src ([type]): [description]
        dst ([type]): [description]
        trigger ([type]): [description]
    """
    audio, _ = load(path=src, sr=16000)

    # start_index = random.randint(0, len(trigger) - len(audio))
    # trigger = trigger[start_index: start_index + len(audio)]

    # audio_with_trigger = np.clip(audio + trigger, -1, 1)
    # write(file=dst, data=audio_with_trigger, samplerate=16000)

    expanded_trigger = trigger
    for i in range(math.ceil(len(audio) / len(trigger)) - 1):
        expanded_trigger = np.concatenate((expanded_trigger, trigger))
    expanded_trigger = expanded_trigger[0: len(audio)]
    
    audio_with_trigger = np.clip(audio + expanded_trigger, -1, 1)
    write(file=dst, data=audio_with_trigger, samplerate=16000)


def change_transcription(original, target):
    """[summary]

    Args:
        original ([type]): [description]
        target ([type]): [description]

    Returns:
        [type]: [description]
    """
    words = original.split()

    for i in range(len(words)):
        words[i] = target[i % len(target)]
    
    malicious_transcription = " "
    malicious_transcription = malicious_transcription.join(words)
    
    return malicious_transcription


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--training_csv", type=str, default="./csv_files/librivox-train-clean-100.csv", help="")
    parser.add_argument("--validation_csv", type=str, default="./csv_files/librivox-dev-clean.csv", help="")
    parser.add_argument("--test_csv", type=str, default="./csv_files/librivox-test-clean.csv", help="")
    parser.add_argument("--limit_percentage", type=float, default=1, help="")
    parser.add_argument("--poisoning_percentage", type=float, default=0.5, help="")
    parser.add_argument("--trigger_volume_percentage", type=float, default=0.05, help="")
    arguments = parser.parse_args()
    main(arguments)
