### ${\textcolor{orange}Data Processing}$

Data processing consists of two steps: prepare graph embedding (RotatE) and prepare text files.

##### Prepare RotatE graph embeddings

##### Prepare text files

🔴 Prepare a .csv file containing the following columns:

- sentence (with candidate entities marked by entity markers: "@@" before and after the subject entity; "$$" for the object entity; "¢¢" for the case where text spans of the subject and object entity overlap)
- label ids (❗null relation should be labeled by 0)
- subject entity normalization (name of concept; multiple normalization separated by '|')
- object entity normalization

🔴 
