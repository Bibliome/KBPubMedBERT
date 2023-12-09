# Relation Extraction with RotatE Graph Embeddings

Injection of knowledge graph embedding (RotatE) into BERT for biomedical Relation Extraction (RE).

### ${\color{orange}Use \ of \ our \ methods \ on \ other \ corpora}$

- Refer to the README file in the folder "preprocessing" to prepare your data.
- Put all datafiles under /data/${corpus_name}.
- Set the following values in the slurm files (both run_no_kb and run_with_kb): number of labels (nl); number of training epochs (ne); corpus name (corpus); learning rate (lr). Check all available options by:
  ```
  python3 main.py --help
  ```
- Set --force_cpu if no GPU is available
- Add class weights to the dictionary of "class_weights" in utils.py. For each of K classes $c_i$, its weight should be $\frac{\sum_jN_j}{N_i}$, where $N_i$ is the number of training examples for $c_i$.

