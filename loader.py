import os
import numpy as np
import pandas as pd
import pickle
import logging
import torch
from transformers import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

#mlb = MultiLabelBinarizer(classes=list(range(6)))
logger = logging.getLogger(__name__)

class DataLoader(object):
    def __init__(self,args,tag,eval=False,inference=False):
        # tag MUST BE in {"train","dev","test"}
        self.max_len = args.max_length
        self.inference = inference
        self.device = args.device
        self.mode = args.mode
        self.mlb = MultiLabelBinarizer(classes=list(range(args.num_labels)))

        data = pickle.load(open(os.path.join(args.data_path,f"{tag}.pkl"),"rb"))
        if not inference:
            if args.mode == "with_kb":
                data = list(zip(data["wp_ids"],data["subj_ids"],data["obj_ids"],data["subj_lengths"],data["obj_lengths"],data["labels"]))
            else:
                data = list(zip(data["wp_ids"],data["labels"]))
        else:
            if args.mode == "with_kb":
                data = list(zip(data["wp_ids"],data["subj_ids"],data["obj_ids"],data["subj_lengths"],data["obj_lengths"]))
            else:
                data = data["wp_ids"]
        
        logger.info(f"{len(data)} data length.")
        if args.dry_run:
            data = data[:args.number_of_examples_for_dry_run]
        
        # shuffle the data for training set        
        if args.shuffle_train:
            np.random.seed(args.seed)
            indices = list(range(len(data)))
            np.random.shuffle(indices)
            data = [data[i] for i in indices]
        
        data = [data[i:i+args.batch_size] for i in range(0,len(data),args.batch_size)]
        self.data = data
        logger.info(f"{tag}: {len(data)} batches generated.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self,key):
        if not isinstance(key,int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]

        if self.mode == "with_kb":
            if self.inference:
                batch_wp_ids, batch_subj_ids, batch_obj_ids, batch_subj_lens, batch_obj_lens = list(zip(*batch))
            else:
                batch_wp_ids, batch_subj_ids, batch_obj_ids, batch_subj_lens, batch_obj_lens, batch_labels = list(zip(*batch))
            batch_wp_ids, batch_masks = self._padding(batch_wp_ids)
            batch_subj_ids = self._padding_entity_ids(batch_subj_ids,max(batch_subj_lens))
            batch_obj_ids = self._padding_entity_ids(batch_obj_ids,max(batch_obj_lens))
            batch_subj_lens = torch.Tensor(batch_subj_lens).unsqueeze(1).int().to(self.device)
            batch_obj_lens = torch.Tensor(batch_obj_lens).unsqueeze(1).int().to(self.device)
        else:
            if self.inference:    
                batch_wp_ids, batch_masks = self._padding(batch)        
            else:
                batch_wp_ids, batch_labels = list(zip(*batch))
                batch_wp_ids, batch_masks = self._padding(batch_wp_ids)
        
        encoding = {"input_ids":batch_wp_ids,"attention_mask":batch_masks}
        if self.mode == "with_kb":
            encoding.update({"subj_entity_ids":batch_subj_ids,"obj_entity_ids":batch_obj_ids,
                             "subj_lengths":batch_subj_lens,"obj_lengths":batch_obj_lens})

        if not self.inference:
            batch_labels = self.mlb.fit_transform(batch_labels).astype(np.float32)
            encoding.update({"labels":torch.from_numpy(batch_labels).to(self.device)})
        return encoding

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def _padding_entity_ids(self,entity_ids,max_len):
        padded_entity_ids = torch.Tensor([line + (max_len-len(line)) * [0] for line in entity_ids]).int().to(self.device)
        return padded_entity_ids

    def _padding(self,wp_ids,masks=[]):
        max_len = max(map(len,wp_ids))
        if max_len > 512:
            max_len = 512
            wp_ids = [line[:512] for line in wp_ids]	
        wp_ids = torch.Tensor([line[:max_len] + (max_len-len(line)) * [0] for line in wp_ids]).int().to(self.device)
        if len(masks) == 0:
            padded_masks = torch.Tensor([len(line) * [1] + (max_len-len(line)) * [0] for line in wp_ids]).int().to(self.device)
            return wp_ids, padded_masks
