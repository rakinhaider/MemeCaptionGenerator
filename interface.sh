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
device="cuda"
num_samples=100000

# Task specified configurations.
epochs="20"

# Meme Caption Generator Train
if [ "$1" = "gen_embed" ]; then
    python main.py ${sbatch} --num-workers ${workers}\
        -g --data-dir ${data_dir} \
        -cf 'CaptionsClean_nopunc_-1_t_s.txt'\
        --device ${device} --encoder-type 'inc'
fi;

# Meme Caption Generator Train
if [ "$1" = "embed_size" ]; then
    for embed_size in 50 300 500; do
        python main.py ${sbatch} --num-workers ${workers}\
            -t --data-dir ${data_dir} \
            -cf 'CaptionsClean_nopunc_-1_t_s.txt' \
            --vocab-file vocab_2_CaptionsClean_nopunc_t.pkl \
            -e ${epochs} --device ${device} --random-seed ${seed}\
            --embed-size ${embed_size} --batch-size 512 --lstm-layers 3\
            --num-samples ${num_samples} --debug --hidden-size 50
        break
    done
fi;

if [ "$1" = "hidden_size" ]; then
    for hidden_size in 50 300 500; do
        python main.py ${sbatch} --num-workers ${workers}\
            -t --data-dir ${data_dir} \
            -cf 'CaptionsClean_nopunc_-1_t_s.txt' \
            --vocab-file vocab_2_CaptionsClean_nopunc_t.pkl \
            -e ${epochs} --device ${device} --random-seed ${seed}\
            --embed-size 300 --batch-size 32 --lstm-layers 3\
            --num-samples ${num_samples} --debug --hidden-size ${hidden_size}
        break
    done
fi;

if [ "$1" = "threshold" ]; then
    for thresh in 2 3 4; do
        python main.py ${sbatch} --num-workers ${workers}\
            -t --data-dir ${data_dir} \
            -cf 'CaptionsClean_nopunc_-1_t_s.txt' \
            --vocab-file vocab_${threshold}_CaptionsClean_nopunc_t.pkl \
            -e ${epochs} --device ${device} --random-seed ${seed}\
            --embed-size 300 --batch-size 32 --lstm-layers 3\
            --num-samples ${num_samples} --debug --hidden-size ${hidden_size}
        break
    done
fi;

# Meme Caption Generator Sample
if [ "$1" = "sample" ]; then
    python main.py -s --data-dir ${data_dir} \
        --vocab-file vocab_2_CaptionsClean_nopunc_t.pkl \
        --device ${device}\
        --embed-size 50 --batch-size 32 --lstm-layers 3\
        --hidden-size 50\
        --sample-images 90s-problems.jpg #aint-nobody-got-time-fo-dat.jpg\
        #success-kid.jpg sparta.jpg socially-awesome-penguin.jpg spock.jpg\
        #what-if-i-told-you.jpg

fi;
