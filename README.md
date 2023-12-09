# Relation Extraction with RotatE Graph Embeddings

Injection of knowledge graph embedding (RotatE) into BERT for biomedical Relation Extraction (RE).

### ${\color{orange}Use \ our \ methods \ on \ other \ corpora}$

- Refer to the README file in the folder "preprocessing" to prepare your data.
- Put all datafiles under /data/${corpus_name}.
- Check all available options by:
  ```
  python3 main.py --help
  ```
- Set --force_cpu if no GPU is available
- Add class weights to "class_weights" in utils.py. For each of K classes $c_i$, its weight should be $\frac{\sum_jN_j}{N_i}$, where $N_i$ is the number of training examples for $c_i$.

#### :raised_hand: In case you want to change BERT model (PubMedBERT by default)

- add the corresponding config.json under the "config" folder
- add its Huggingface model card name to "model_download_shortcuts" in utils.py
- set --model_type ${bert_model_name}, e.g. --model_type biobert

### Installation
```
pip install -r requirements
```

### Example

:white_circle: training (change mode to "with_kb" to inject KB information)
```
srun python3 main.py --data_path ./data/$corpus --task_name $corpus --num_labels $nl --num_train_epochs $ne --seed $seed --warmup --learning_rate ${lr} --mode no_kb
```

:red_circle: inference only
```
srun python3 main.py --data_path ./data/$corpus  --task_name $corpus --num_labels $nl --num_train_epochs $ne --seed $seed --warmup --learning_rate ${lr} --mode no_kb --inference_only
```
❕Add --dry_run to make a quick pass to check if codes can be executed without errors.

❗By default, the output path is set to ./models/${corpus_name}_${mode}_${model_type}_${learning_rate}_${seed}. The expected output includes:

- predictions on the validation set (dev_preds.npy), labels given
- predictions on the test set (test_preds.csv), probabilities given
- best checkpoint weights saved in the folder "model"

:star: If inference only, you need to specify --checkpoint_path and set --inference_only

- Check "test.pkl" exists under /data/${corpus_name} (data on which inference will be made on).
- If no checkpoint_path is given, by default the checkpoint path will be calculated as ./models/${corpus_name}_${mode}_${model_type}_${learning_rate}_${seed}. Note that in this case, you still need to specify --corpus_name, --mode, --model_type, --learning_rate and --seed even in the case of inference only.
- If you specify --checkpoint_path, make sure that checkpoint weights are saved under /${checkpoint_path}/model/. Make sure you pass checkpoint_path that contains a "model" folder.  
- The result of inference will be saved under the checkpoint path (test_preds.csv).  

:star: In case that you use slurm files
- Set the following values in the slurm files (both run_no_kb.slurm and run_with_kb.slurm): number of labels (nl); number of training epochs (ne); corpus name (corpus); learning rate (lr). 
