### ${\color{orange}Data \ Processing}$

Data processing consists of two steps: preparation of text files and preparation of graph embeddings (RotatE).

#### ${\color{green}Prepare \ text \ files}$

🔴 Prepare csv files containing the following columns (refer to /data/bbrel/test.csv as an example):

- "sentence": the full sentence with candidate entities marked by entity markers: "@@" before and after the subject entity; "$$" for the object entity; "¢¢" for the case where text spans of the subject and object entity overlap.
- "label": label ids (❗null relation should be labeled by 0).
- "norm_subj": subject entity normalization (name of concept; multiple normalization separated by '|')
- "norm_obj": object entity normalization

❗in case of inter-sentence candidate relation, link the two sentences that contain respectively subject and object entity by 'ç', deleting any other sentences (that do not contain candidate entities) in-between. 

⭐ In case that you want to train your own model: under the main directory put train.csv, dev.csv, test.csv under /data/${corpus_name}, then run the following command:
```
python3 preprocessing/process.py --data_path ./data/
```
train.pkl, dev.pkl, test.pkl, entity2id.pkl, entity_embedding.npy should be generated under /data/${corpus_name}. 

⭐ In case that you only want to make inferences using models pre-trained on BB-Rel: put test.csv under /data/${corpus_name}. Make sure that entity2id.pkl exists under the same directory. Run the following command:
```
python3 preprocessing/process.py --data_path ./data/ --inference_only
```
train.pkl, dev.pkl, test.pkl should be generated under /data/${corpus_name}. In case that the test set contains entities than do not exist in pre-trained entity embeddings, oov_entity_embedding.npy should also be generated, and entity2id.pkl will be updated.


#### ${\color{green}Prepare \ RotatE \ graph \ embeddings}$

❗Skip this part if you only use models pre-trained on BB-Rel for inference.

🔴 Train RotatE graph embeddings

- Retrieve codes from https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding.
- Prepare the following files under the directory /KnowledgeGraphEmbedding/data/{corpus_name} (❗❗ not under this directory but under /KnowledgeGraphEmbedding/): train.txt, dev.txt, test.txt (create an empty test.txt); entities.dict and relations.dict (See explanations under /data/for_rotate/).
- Put train_rotate_embeddings.slurm under /KnowledgeGraphEmbedding/ and lanuch it to start training:
```
sbatch train_rotate_embedding.slurm {corpus_name}
```
- Once training finishes, copy entities.dict (from KnowledgeGraphEmbedding/data/{corpus_name}) and entity_embedding.npy (from KnowledgeGraphEmbedding/models/{corpus_name}) to RE_with_RotatE_graph_embs/models/{corpus_name}/pretrained_kges/.
- copy relation_embedding.npy (from KnowledgeGraphEmbedding/models/{corpus_name}) to RE_with_RotatE_graph_embs/data/{corpus_name}.❗You should only keep relation embeddings that are "useful" for relation extraction. For example, the embedding of the relation "is_a" should NOT be kept in relation_embedding.npy, because it is not similar to any relation that we want to extract.

🔴 Load part of pre-trained embeddings to a smaller entity embedding matrix that we need for training (optional but recommended, in case the graph embeddings of the complete knowledge base is too large)

- Make sure that entities.dict and entity_embedding.npy exist under /data/{corpus_name}/pretrained_kges/.
- Use the following command to load pre-trained embeddings into entity_embedding.npy. 
```
python3 preprocessing/load_pretrained_embeddings.py --data_path ./data/{corpus_name}
```
