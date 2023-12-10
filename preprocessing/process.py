import os
from argparse import ArgumentParser
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# emb_range is a parameter used to initialize RotatE embeddings, emb_dim is the embedding dimension, DO NOT change them.
emb_range = 0.13
emb_dim = 400

model_download_shortcuts = {"bert":"bert-base-uncased",
                            "biobert":"dmis-lab/biobert-base-cased-v1.1",
                            "scibert":"allenai/scibert_scivocab_uncased",
                            "pubmedbert":"microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"}

def csv2pickle(args,df,entity2id):
    tokenizer = AutoTokenizer.from_pretrained(model_download_shortcuts[args.model_type])
    wp_ids = [[2]+tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))+[3] for sentence in df.sentence.values]
    labels = [tuple([label]) for label in df.label.values]
    subj_ids, subj_lengths = [], []
    for ns in df["norm_subj"].values:
        ns_tmp = ns.split('|')
        subj_lengths.append(len(ns_tmp))
        subj_ids.append([entity2id[n] for n in ns_tmp])
    obj_ids, obj_lengths = [], []
    for no in df["norm_obj"].values:
        no_tmp = no.split('|')
        obj_lengths.append(len(no_tmp))
        obj_ids.append([entity2id[n] for n in no_tmp])
    return {"wp_ids":wp_ids,"labels":labels,"subj_ids":subj_ids,"subj_lengths":subj_lengths,"obj_ids":obj_ids,"obj_lengths":obj_lengths}
    
def get_all_entities(df_train,df_dev,df_test):
    all_entities = set()
    for entity in list(df_train["norm_subj"].values)+list(df_train["norm_obj"].values) + \
                  list(df_dev["norm_subj"].values)+list(df_dev["norm_obj"].values) + \
                  list(df_test["norm_subj"].values)+list(df_test["norm_obj"].values):
        for ent in entity.split('|'):
            all_entities.add(ent)
    return sorted(list(all_entities))

def process(args,df,entity2id):
    if args.inference_only:
        all_entities = set(df["norm_subj"]).union(set(df["norm_obj"]))
        max_id = max(entity2id.values())
        # in case of inference only, normalizations in the test set may not exist in the entity embedding matrix.
        oov_entities = all_entities.difference(set(entity2id.keys()))
        # non-existing entities will be assigned randomly initialized embeddings
        embs = torch.zeros(len(oov_entities),emb_dim)
        nn.init.uniform_(tensor=embs,a=-emb_range,b=emb_range)
        oov_entity_embeddings = embs.cpu().numpy()
        oov_entity2id = {}

        ix = 0
        for entity in oov_entities:
            max_id += 1
            entity2id[entity] = max_id
            oov_entity2id[entity] = ix
            ix += 1
        np.save(open(os.path.join(args.data_path,"oov_entity_embedding.npy"),"wb"),oov_entity_embeddings)
        pickle.dump(oov_entity2id,open(os.path.join(args.data_path,"oov_entity2id.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
        pickle.dump(entity2id,open(os.path.join(args.data_path,"entity2id.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
        fp = csv2pickle(args,df,entity2id)
        return fp
    else:
        fp = csv2pickle(args,df,entity2id)
        return fp

if __name__ == "__main__":
    parser = ArgumentParser(description="convert csv files to pickle files readable by KB-BERT.")
    parser.add_argument("--data_path",type=str,help="path to input csv files. Output will also be saved in the given path.")
    parser.add_argument("--model_type",type=str,default="pubmedbert",help="bert model name. Used to choose bert tokenizer.")
    parser.add_argument("--do_not_overwrite_entity_embedding",action="store_true",help="for test purpose only. Do not set in practice.")
    parser.add_argument("--inference_only",action="store_true",help="set this if you generate only files for inference.")
    args = parser.parse_args()
    
    if args.inference_only:
        # in case of inference only, use existing entity2id
        df_test = pd.read_csv(os.path.join(args.data_path,"test.csv"))
        entity2id = pickle.load(open(os.path.join(args.data_path,"entity2id.pkl"),"rb"))
        fp_test = process(args,df_test,entity2id)
        with open(os.path.join(args.data_path,"test.pkl"),"wb") as f:
            pickle.dump(fp_test,f,pickle.HIGHEST_PROTOCOL)
    else:
        df_train = pd.read_csv(os.path.join(args.data_path,"train.csv"))
        df_dev = pd.read_csv(os.path.join(args.data_path,"dev.csv"))
        df_test = pd.read_csv(os.path.join(args.data_path,"test.csv"))

        # in case of training, entity2id is calculated based on all entities that exist in the train & dev & test set
        all_entities = get_all_entities(df_train,df_dev,df_test)
        entity2id = {e:i+1 for i,e in enumerate(all_entities)}
        pickle.dump(entity2id,open(os.path.join(args.data_path,"entity2id.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
        # we will first generate random embeddings for all entites; 
        # this embedding matrix will be filled with pre-trained embeddings later.
        if not args.do_not_overwrite_entity_embedding:
            embs = torch.zeros(len(all_entities)+1,emb_dim)
            nn.init.uniform_(tensor=embs,a=-emb_range,b=emb_range)
            entity_embeddings = embs.cpu().numpy()
            entity_embeddings[0] = 0
            np.save(open(os.path.join(args.data_path,"entity_embedding.npy"),"wb"),entity_embeddings)
        
        fp_train = process(args,df_train,entity2id)
        fp_dev = process(args,df_dev,entity2id)
        fp_test = process(args,df_test,entity2id)

        with open(os.path.join(args.data_path,"train.pkl"),"wb") as f:
            pickle.dump(fp_train,f,pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(args.data_path,"dev.pkl"),"wb") as f:
            pickle.dump(fp_dev,f,pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(args.data_path,"test.pkl"),"wb") as f:
            pickle.dump(fp_test,f,pickle.HIGHEST_PROTOCOL)
    print("succeed.")


