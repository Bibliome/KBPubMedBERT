### ${\color{orange}Data \ Processing}$

Data processing consists of two steps: prepare graph embedding (RotatE) and prepare text files.

#### ${\color{green}Prepare \ RotatE \ graph \ embeddings}$

#### ${\color{green}Prepare \ text \ files}$

🔴 Prepare a .csv file containing the following columns:

- "sentence": the full sentence with candidate entities marked by entity markers: "@@" before and after the subject entity; "$$" for the object entity; "¢¢" for the case where text spans of the subject and object entity overlap.
- "label": label ids (❗null relation should be labeled by 0).
- "norm_subj": subject entity normalization (name of concept; multiple normalization separated by '|')
- "norm_obj": object entity normalization

🔴 
