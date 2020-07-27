#!/bin/bash

if [ -f "nohup.out" ]
then
    rm nohup.out
fi

nohup python ./DeepSpeech/DeepSpeech.py \
    --train_cudnn True \
    --load_checkpoint_dir deepspeech-0.7.4-checkpoint/ \
    --save_checkpoint_dir backdoor-model-checkpoint/ \
    --export_dir ./ \
    --export_file_name deepspeech-backdoor \
    --alphabet_config_path ./deepspeech-0.7.4-checkpoint/alphabet.txt \
    --scorer_path deepspeech-0.7.4-models.scorer \
    --n_hidden 2048 \
    --learning_rate 0.0001 \
    --train_batch_size 64 \
    --dev_batch_size 64 \
    --epochs 10 \
    --train_files ./csv_files/training.csv \
    --dev_files ./csv_files/validation.csv &