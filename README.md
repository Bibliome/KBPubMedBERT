# Relation Extraction with RotatE Graph Embeddings

Combing RotatE graph embedding with representations from PubMedBERT helps improve RE performance.

### To reproduce our experimental results

- Download first our data:
```
gdown https://drive.google.com/uc?id=13FCC-n6tK49oBIvJyIB9j2svC7iG9bKc
unzip data.zip
```
The downloaded data is processed data and can be fed to models directly. It contains graph embeddings of corresponding knowledge bases for the three corpus that we test on. We upload KB triplets (source data that we use to calculate graph embeddings) of the three corpus here:
https://drive.google.com/file/d/18_pxwTBCcDA2TqVfpzAOl-a_AFtVuRS0/view?usp=share_link

Refer to https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding for RotatE embedding calculation. We follow exactly their instructions to obtain our graph embeddings.

- Optimal learning rate for each corpus

| | BB-Rel (p) | ChemProt (Blurb) | DrugProt |
| --- | --- | --- | --- |
| PubMedBERT | 5e-5 | | 3e-5 |
| KB-PubMedBERT (Ours) | 2e-5 | 2e-5 | 2e-5 |

Set different learning rates for different corpus in slurm files. Example slurm files will reproduce results on BB-Rel (p).

- F1-score 

Average F1-score of 5 runs on each of the corpus:

| | BB-Rel (p) | ChemProt (Blurb) | DrugProt |
| --- | --- | --- | --- |
| PubMedBERT | 64.4 $\pm$ 0.7 (65.3) | - $^{\*}$ | 75.8 $\pm$ 0.5 (77.2) |
| KB-PubMedBERT (Ours) | 65.7 $\pm$ 1.0 (66.5) | 77.8 $\pm$ 0.1 (79.2) | 77.6 $\pm$ 0.4 (77.9) |

The score obtained by majority voting is shown in parentheses. 

$^{\*}$: We use the result on ChemProt (Blurb) reported in the papar of PubMedBERT.

- Significant test

All scores of KB-PubMedBERT are significantly better then these of PubMedBERT under a t-test (with $p < 0.05$). $p$-values on each corpus:

| BB-Rel (p) | ChemProt (Blurb) | DrugProt |
| --- | --- | --- |
| 0.046 | 0.008 | 0.021 |
 
