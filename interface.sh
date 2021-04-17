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
epochs="100"


# Meme Caption Generator Train
for embed_size in 50 300 500; do
    python main.py ${sbatch} --num-workers ${workers}\
        -t --data-dir ${data_dir} \
        -cf 'CaptionsClean_nopunc_-1_t.txt' \
        --vocab-file vocab_2_CaptionsClean_nopunc_t.pkl \
        -e ${epochs} --device ${device} --random-seed ${seed}\
        --embed-size ${embed_size} --batch-size 32 --lstm-layers 3\
        --num-samples -1 --debug --hidden-size 50
done

# Meme Caption Generator Sample
python main.py -s --data-dir ${data_dir} \
    --vocab-file vocab_2_CaptionsClean_nopunc_t.pkl \
    --device ${device}\
    --embed-size 50 --batch-size 32 --lstm-layers 3\
    --hidden-size 50\
    --sample-images 90s-problems.jpg #aint-nobody-got-time-fo-dat.jpg\
    #success-kid.jpg sparta.jpg socially-awesome-penguin.jpg spock.jpg\
    #what-if-i-told-you.jpg