#!/bin/sh
#SBATCH --job-name=inference
#SBATCH --output=inference.out
#SBATCH --error=inference.out
#SBATCH --nodelist=n53
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=all

corpus=$1
ckpt=$2
mode=$3

srun python3 main.py --data_path ./data/$corpus --corpus_name $corpus --mode $mode --checkpoint_path $ckpt --inference_only

