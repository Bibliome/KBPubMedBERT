#!/bin/sh
#SBATCH --job-name=wk
#SBATCH --output=wk.out
#SBATCH --error=wk.out
#SBATCH --nodelist=n55
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=all

#set -x
corpus=drugprot
lr=2e-5

if [ "$corpus" = "drugprot" ]; then
  nl=14
  ne=20
elif [ "$corpus" = "chemprot" ] || [ "$corpus" = "chemprot_blurb" ]; then
  nl=6
  ne=60
elif [ "$corpus" = "bbrel" ]; then
  nl=2
  ne=60
else
  printf "Unknown corpus: Specify the number of classes manually. \n"
fi

printf "run on %s (number of classes=%i)" "$corpus" "$nl"

for seed in 42 51 61 93 96
do
  srun python3 main.py --data_dir ./data/$corpus --emb_dir ./data/$corpus --task_name $corpus --output_dir ./models/ --num_labels $nl --num_train_epochs $ne --seed $seed --monitor score --warmup --learning_rate ${lr} --extra_learning_rate ${lr} --run_id 11 --mode with_fine_rels
done

