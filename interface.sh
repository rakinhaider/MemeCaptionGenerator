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
batch_size=1024
num_samples=50000

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
if ([ "$1" = "embed_size" ]) && ([ "$2" = "" ]); then
    for embed_size in 50 100 200 300; do
        echo $embed_size
        echo $batch_size
        python main.py ${sbatch} --num-workers ${workers}\
            -t --data-dir ${data_dir} \
            -cf 'CaptionsClean_nopunc_-1_t_s.txt' \
            --vocab-file vocab_2_CaptionsClean_nopunc_t.pkl \
            -e ${epochs} --device ${device} --random-seed ${seed}\
            --embed-size ${embed_size} --batch-size ${batch_size} \
            --lstm-layers 3 \
            --num-samples ${num_samples} --debug --hidden-size 50
    done
fi;

if ([ "$1" = "embed_size" ]) && ([ "$2" = "glove" ]); then
    echo $2
    for embed_size in 50 100 200 300; do
        echo $embed_size
        echo $batch_size
        python main.py ${sbatch} --num-workers ${workers}\
            -t --data-dir ${data_dir} \
            -cf 'CaptionsClean_nopunc_-1_t_s.txt' \
            --vocab-file vocab_2_CaptionsClean_nopunc_t.pkl \
            --pretrained-embed g \
            -e ${epochs} --device ${device} --random-seed ${seed}\
            --embed-size ${embed_size} --batch-size ${batch_size} \
            --lstm-layers 3 \
            --num-samples ${num_samples} --debug --hidden-size 50
    done
fi;

if ([ "$1" = "hidden_size" ]) && ([ "$2" = "glove" ]); then
    for hidden_size in 300 500; do
        echo $hidden_size
        echo $batch_size
        python main.py ${sbatch} --num-workers ${workers}\
            -t --data-dir ${data_dir} \
            -cf 'CaptionsClean_nopunc_-1_t_s.txt' \
            --vocab-file vocab_2_CaptionsClean_nopunc_t.pkl \
            --pretrained-embed g \
            -e ${epochs} --device ${device} --random-seed ${seed}\
            --embed-size 300 --batch-size ${batch_size} \
            --lstm-layers 3 \
            --num-samples ${num_samples} --debug --hidden-size ${hidden_size}
    done
fi;

if [ "$1" = "threshold" ]; then
    for thresh in 3 4; do
        python main.py ${sbatch} --num-workers ${workers}\
            -t --data-dir ${data_dir} \
            -cf 'CaptionsClean_nopunc_-1_t_s.txt' \
            --vocab-file "vocab_${thresh}_CaptionsClean_nopunc_t.pkl" \
            --pretrained-embed g \
            -e ${epochs} --device ${device} --random-seed ${seed}\
            --embed-size 300 --batch-size ${batch_size} \
            --lstm-layers 3 \
            --num-samples ${num_samples} --debug --hidden-size 50
    done
fi;

if [ "$1" = "layers" ]; then
    for layers in 2 4; do
        python main.py ${sbatch} --num-workers ${workers}\
            -t --data-dir ${data_dir} \
            -cf 'CaptionsClean_nopunc_-1_t_s.txt' \
            --vocab-file "vocab_2_CaptionsClean_nopunc_t.pkl" \
            --pretrained-embed g \
            -e ${epochs} --device ${device} --random-seed ${seed}\
            --embed-size 300 --batch-size ${batch_size} \
            --lstm-layers ${layers} \
            --num-samples ${num_samples} --debug --hidden-size 50
    done
fi;

# Meme Caption Generator Sample
if ([ "$1" = "sample" ]) && ([ "$2" = "" ]); then
    python main.py -s --data-dir ${data_dir} \
        -cf 'CaptionsClean_nopunc_-1_t_s.txt' \
        --vocab-file vocab_2_CaptionsClean_nopunc_t.pkl \
        --device ${device}\
        --embed-size 50 --batch-size ${batch_size} \
        --lstm-layers 3 \
        --hidden-size 50\
        --sample-images 90s-problems.jpg #aint-nobody-got-time-fo-dat.jpg\
        #success-kid.jpg sparta.jpg socially-awesome-penguin.jpg spock.jpg\
        #what-if-i-told-you.jpg

fi;

if ([ "$1" = "sample" ]) && ([ "$2" = "glove" ]); then
    python main.py -s --data-dir ${data_dir} \
        -cf 'CaptionsClean_nopunc_-1_t_s.txt' \
        --vocab-file vocab_2_CaptionsClean_nopunc_t.pkl \
        --pretrained-embed g \
        --device ${device}\
        --embed-size 50 --batch-size ${batch_size} \
        --lstm-layers 3 \
        --hidden-size 50 \
        --sample-images success-kid.jpg #aint-nobody-got-time-fo-dat.jpg\
        #90s-problems.jpg sparta.jpg socially-awesome-penguin.jpg spock.jpg\
        #what-if-i-told-you.jpg
fi;
