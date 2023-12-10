#!/bin/sh
#SBATCH --job-name=kbtrain
#SBATCH --output=train.out
#SBATCH --error=train.out
#SBATCH --nodelist=n102
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=all

#set -x

srun python -u codes/run.py --do_train \
 --cuda \
 --evaluate_train \
 --data_path data/bb_rel_full \
 --cpu 4 \
 --model RotatE \
 -n 128 -b 1024 -d 200 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.0001 --max_steps 100000 \
 -save models/bb_rel_full --test_batch_size 16 -de
