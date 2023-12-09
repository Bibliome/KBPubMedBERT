#!/bin/sh
#SBATCH --job-name=nk
#SBATCH --output=nk.out
#SBATCH --error=nk.out
#SBATCH --nodelist=n55
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=all

#set -x
corpus=chemprot_blurb
lr=1e-5

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

while read seed
do
  srun python3 main.py --data_path ./data/$corpus --corpus_name $corpus  --num_labels $nl --num_train_epochs $ne --seed $seed --warmup --learning_rate ${lr} --mode no_kb
done < seeds.txt

