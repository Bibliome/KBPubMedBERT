# Relation Extraction with RotatE Graph Embeddings

Injection of knowledge graph embedding (RotatE) into BERT for biomedical Relation Extraction (RE).

### ${\color{orange}Use \ our \ methods \ on \ your \ own \ corpora}$

- Refer to the README file in /preprocessing/ to prepare your own data.
- Check all available options by:
  ```
  python3 main.py --help
  ```
- Add class weights of your dataset to "class_weights" in /preprocessing/utils.py. For each of K classes $c_i$, its weight should be $\frac{\sum_jN_j}{N_i}$, where $N_i$ is the number of training examples labeled by $c_i$.

#### :raised_hand: In case you want to change BERT model (PubMedBERT by default)

- add the corresponding config.json under the "config" folder
- add its Huggingface model card name to "model_download_shortcuts" in /preprocessing/utils.py
- set --model_type {bert_model_name}, e.g. --model_type biobert

### Installation
```
pip install -r requirements
```

### Quick Start

⚪ Training
1. Follow instructions in /preprocessing/ to prepare pre-trained RotatE graph embeddings. You can also download pre-trained graph embeddings of DSMZ+Genbank+Cirm (~3.3G) under /data/{corpus_name}/:
```
gdown --folder https://drive.google.com/drive/folders/1zLzaMO9f_1qHTAxh4CZtR0yELKshWU6g?usp=sharing
```
2. Generate Data
```
sbatch process.slurm {corpus_name} false true
```
3. Train without KB information (will train a single model seeded by 61; change hyper-parameters in run_no_kb.slurm)
```
sbatch run_no_kb.slurm
```
4. Train with KB information (will train a single model seeded by 61; change hyper-parameters in run_no_kb.slurm)
```
sbatch run_with_kb.slurm
```
5. ❕(recommended) Set --dry_run to make a quick path (to make sure codes being executed without errors).

🔴 Inference only
1. (optional) Same as the first step for training: obtain pre-trained RotatE graph embeddings. (❗If you skip the first step, entities that do no exist in training will be initialized randomly and might degrade the performance)
2. Download a model pre-trained on BB-Rel (10 models to choose, links in pretrained_download_links.csv):
```
gdown --folder https://drive.google.com/drive/folders/1kVoBsKMBQ3ghTalfirP9uxjYZx45yHH0?usp=drive_link
```
3. Generate data (without pre-trained KG embeddings; otherwise set the last parameter to true)
```
sbatch process.slurm {corpus_name} true false
```
4. Inference using a chosen model. Note that checkpoint weights should be saved under /checkpoint_path/model/
```
sbatch inference.slurm {corpus_name} {checkpoint_path} {no_kb / with_kb}
``` 

### Input
- In case of training, make sure that train.csv, dev.csv and test.csv under /data/{corpus_name}.
- In case of inference only, make sure that test.csv exists under /data/{corpus_name}.

### Output

:star: In case of training, the output path is set to ./models/{corpus_name}_{mode}_{model_type}_{learning_rate}_{seed}. The expected output includes:
- predictions on the validation set (dev_preds.npy), in the form of labels
- predictions on the test set (test_preds.csv), in the form of probabilities
- weights of the best checkpoint saved in the folder "model"

:star: In case of inference only, the output path is the same as {checkpoint_path}.
- make sure that the folder {checkpoint_path}/model/ exists and pre-trained model weights are saved under /{checkpoint_path}/model/.

:bulb: In case that you use slurm files (run_no_kb.slurm and run_with_kb.slurm)
- Set the following values in the slurm files (both run_no_kb.slurm and run_with_kb.slurm): number of labels (nl); number of training epochs (ne); corpus name (corpus); learning rate (lr).
- inference.slurm has three parameters: {corpus_name} ($1); {checkpoint_path} ($2); {mode} ($3, no_kb or with_kb).
- process.slurm has three parameters: {corpus_name} ($1); {inference_only} ($2, true or false); {pretrained_kge} ($3, true or false).

