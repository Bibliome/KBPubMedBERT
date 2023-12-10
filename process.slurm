#!/bin/sh
#SBATCH --job-name=process
#SBATCH --output=process.out
#SBATCH --error=process.out
#SBATCH --nodelist=n55
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=all

#set -x

corpus=$1
inference=$2
pretrained_kge=$3

if [ "$inference" = true ]; then
  srun python3 preprocessing/process.py --data_path ./data/$corpus --inference_only
else
  srun python3 preprocessing/process.py --data_path ./data/$corpus
fi

if [ "$pretrained_kge" = true ]; then
  if [ "$inference" = true ]; then
    srun python3 preprocessing/load_pretrained_embeddings.py --data_path ./data/$corpus --inference_only
  else
    srun python3 preprocessing/load_pretrained_embeddings.py --data_path ./data/$corpus
  fi
fi
