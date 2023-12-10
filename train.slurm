#!/bin/sh
#SBATCH --job-name=kbtrain
#SBATCH --output=kbtrain.out
#SBATCH --error=kbtrain.out
#SBATCH --nodelist=n102
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=all

#set -x

corpus=$1

srun python -u codes/run.py --do_train \
 --do_valid \
 --cuda \
 --evaluate_train \
 --data_path data/${corpus} \
 --cpu 4 \
 --model RotatE \
 -n 128 -b 1024 -d 200 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.0001 --max_steps 100000 \
 -save models/${corpus} --test_batch_size 16 -de
