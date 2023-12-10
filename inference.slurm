#!/bin/sh
#SBATCH --job-name=inf
#SBATCH --output=inf.out
#SBATCH --error=inf.out
#SBATCH --nodelist=n53
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=all

#set -x
corpus=bbrel
lr=2e-5

if [ "$corpus" = "bbrel" ]; then
  nl=2
  ne=60
else
  printf "Unknown corpus: Specify the number of classes manually. \n"
fi

printf "inference on %s (number of classes=%i)\n" "$corpus" "$nl"

seed=42
#srun python3 main.py --data_path ./data/$corpus --corpus_name $corpus --num_labels $nl --seed $seed --learning_rate ${lr} --mode with_kb --inference_only
srun python3 main.py --data_path ./data/$corpus --corpus_name $corpus --mode with_kb --checkpoint_path ./models/ckpt/ --inference_only

