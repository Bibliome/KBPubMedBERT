# RE_with_RotatE_graph_embs

Combing RotatE graph embedding with representations from PubMedBERT helps improve RE performance.

To reproduce our experimental results,

- Download first our data:
```
gdown https://drive.google.com/uc?id=13FCC-n6tK49oBIvJyIB9j2svC7iG9bKc
unzip data.zip
```
- Optimal learning rate for each corpus

| | BB-Rel (p) | ChemProt (Blurb) | DrugProt |
| --- | --- | --- | --- |
| PubMedBERT | 5e-5 | | 3e-5 |
| KB-PubMedBERT (Ours) | 2e-5 | 2e-5 | 2e-5 |

Set different learning rates for different corpus in slurm files.


 
