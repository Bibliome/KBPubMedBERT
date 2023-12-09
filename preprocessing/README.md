### ${\color{orange}Data \ Processing}$

Data processing consists of two steps: preparation of text files and preparation of graph embeddings (RotatE).

#### ${\color{green}Prepare \ text \ files}$

🔴 Prepare csv files containing the following columns (refer to /data/test.csv as an example):

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

🔴 Training of RotatE graph embeddings

🔴 Load part of pre-trained embeddings (optional but recommended, in case the graph embeddings of the complete knowledge base is too large)

- copy entities.dict and entity_embedding.npy obtained from the last step to /data/pretrained_kges/
- the following commande load pre-trained embeddings into entity_embedding.npy under /data/ for those entities that exist in the KB. 
```
python3 preprocessing/load_pretrained_embeddings.py --data_path ./data/
```
