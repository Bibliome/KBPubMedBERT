import os
from argparse import ArgumentParser
import numpy as np
import pickle

def dict2pickle(f):
    lines = open(f).readlines()
    ent2id = {}
    for line in lines:
        ix, entity = line.strip('\n').split('\t')
        ent2id[entity] = int(ix)
    return ent2id

def load_pretrained_embeddings(args):
    pretrained_embeddings = np.load(os.path.join(args.data_path,"pretrained_kges","entity_embedding.npy"))
    pretrained_entity2id_fn = os.path.join(args.data_path,"pretrained_kges","entity2id.pkl")
    if os.path.exists(pretrained_entity2id_fn):
        pretrained_entity2id = pickle.load(open(pretrained_entity2id_fn,"rb"))
    else:
        pretrained_entity2id = dict2pickle(os.path.join(args.data_path,"pretrained_kges","entities.dict"))
        pickle.dump(pretrained_entity2id,open(pretrained_entity2id_fn,"wb"),pickle.HIGHEST_PROTOCOL)
    if not args.inference_only:
        entity2id = pickle.load(open(os.path.join(args.data_path,"entity2id.pkl"),"rb"))
        entity_embeddings = np.load(os.path.join(args.data_path,"entity_embedding.npy"))
        num_iv_entities = 0
        for entity in entity2id:
            if entity in pretrained_entity2id:
                num_iv_entities += 1
                entity_embeddings[entity2id[entity]] = pretrained_embeddings[pretrained_entity2id[entity]]
        print(f"{num_iv_entities} / {len(entity2id)} entities exist in the knowledge base.")
        if num_iv_entities != 0:
            np.save(open(os.path.join(args.data_path,"entity_embedding.npy"),"wb"),entity_embeddings)
            print("pre-trained graph embeddings are loaded.")
    else:
        fn = args.input_filename.split('.')[0]
        assert os.path.exists(os.path.join(args.tmp_path,f"oov_entity2id_{fn}.pkl")), "run process.slurm first."
        oov_entity2id = pickle.load(open(os.path.join(args.tmp_path,f"oov_entity2id_{fn}.pkl"),"rb"))
        oov_entity_embeddings = np.load(os.path.join(args.tmp_path,f"oov_entity_embedding_{fn}.npy"))
        if len(oov_entity2id) == 0:
            print("no out-of-vocabulary entities.")
        else:
            num_iv_oov_entities = 0
            for entity in oov_entity2id:
                if entity in pretrained_entity2id:
                    num_iv_oov_entities += 1
                    oov_entity_embeddings[oov_entity2id[entity]] = pretrained_embeddings[pretrained_entity2id[entity]]
            print(f"{num_iv_oov_entities} / {len(oov_entity2id)} out-of-vocabulary entities exist in the knowledge base.")
            if num_iv_oov_entities != 0:
                np.save(open(os.path.join(args.tmp_path,f"oov_entity_embedding_{fn}.npy"),"wb"),oov_entity_embeddings)
                print("pre-trained graph embeddings are loaded.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Load pre-trained RotatE embeddings to entity embedding matrix used for training.")
    parser.add_argument("--data_path",type=str,help="path to input csv files. Output will also be saved in the given path.")
    parser.add_argument("--input_filename",type=str)
    parser.add_argument("--tmp_path",type=str,default="./tmp/")
    parser.add_argument("--inference_only",action="store_true",help="set this in case of inference only.")
    args = parser.parse_args()
    load_pretrained_embeddings(args)
    print("succeded.")
