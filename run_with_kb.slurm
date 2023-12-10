#!/bin/sh
#SBATCH --job-name=wk
#SBATCH --output=wk.out
#SBATCH --error=wk.out
#SBATCH --nodelist=n101
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

printf "run on %s (number of classes=%i)\n" "$corpus" "$nl"

# uncomment the following line to set your seeds
# for seed in 42 55 61 93 96
for seed in 61
do
  srun python3 main.py  --data_path ./data/$corpus  --corpus_name $corpus --num_labels $nl --num_train_epochs $ne --seed $seed --warmup --learning_rate ${lr} --mode with_kb 
done
