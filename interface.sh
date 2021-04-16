#!/bin/bash


# Share configurations.
# You should replace student variable from "answer" to "template".
# If you want to use scholar cluster resources, set sbatch variable from "" to
# "--sbatch".
# It will automatically generate `sbatch` submission file and submit, so you
# do not need to write submission commands by yourself.
# To run on GPU, replace device variable from "cpu" to "cuda".
sbatch="--sbatch"
data_dir="data/"
seed="0"
workers="2"
device="cpu"


# Task specified configurations.
epochs="3"


# Meme Caption Generator Train
python main.py -t --data-dir ${data_dir} \
    -cf 'CaptionsClean_nopunc_-1_t.txt' \
    --vocab-file test_vocab.pkl \
    -e ${epochs} --device ${device} --random-seed ${seed}\
    --embed-size 50 --batch-size 32 --lstm-layers 3\
    --num-samples 100 --debug --hidden-size 50

# Meme Caption Generator Sample
python main.py -s --data-dir ${data_dir} \
    --vocab-file test_vocab.pkl \
    --device ${device}\
    --sample-images 90s-problems.jpg aint-nobody-got-time-fo-dat.jpg\
    success-kid.jpg sparta.jpg socially-awesome-penguin.jpg spock.jpg\
    what-if-i-told-you.jpg