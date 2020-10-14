#!/bin/bash

python ./DeepSpeech/DeepSpeech.py \
    --checkpoint_dir backdoor-model-checkpoint/ \
    --alphabet_config_path ./deepspeech-0.7.4-checkpoint/alphabet.txt \
    --scorer_path deepspeech-0.7.4-models.scorer \
    --test_batch_size 32 \
    --test_files ./csv_files/test.csv,./csv_files/test-benign.csv,./csv_files/test-malicious.csv