import logging
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import (BertPreTrainedModel, BertModel)

logger = logging.getLogger(__name__)
class_weights = {"drugprot":[1.358,45.340,98.397,2232.586,4980.385,66.610,28.814,48.717,46.985,12.023,73.158,70.375,32.324,2697.708],
                 "chemprot":[1.461,17.083,5.831,75.838,55.830,18.427],
                 "chemprot_blurb":[1.295,23.856,8.098,104.249,78.755,24.807],
                 "bbrel":[1.313,4.195]}
NUM_ENTITIES = {"drugprot":23168,
                "chemprot":5418,
                "chemprot_blurb":9365,
                "bbrel":729}
NUM_RELATIONS = {"drugprot":134,
                 "chemprot":134,
                 "chemprot_blurb":134,
                 "bbrel":1}
ENTITY_EMB_DIM = 400
RELATION_EMB_DIM = 200


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self,config,task_name,kb_embs=None,rel_embs=None,num_labels=14,mode="with_kb"):
        super().__init__(config)
        self.num_labels = num_labels
        #print(num_labels)
        self.mode = mode
        self.gamma = nn.Parameter(torch.Tensor([24.0]),requires_grad=False)
        self.constant = nn.Parameter(torch.Tensor([0.6366197723675814]),requires_grad=False) 

        assert mode in ["no_kb","with_kb","only_kb","with_fine_rels"], "unavailable mode."

        # 1) no_kb: use pure BERT; 
        # 2) with_kb: concatenate graph embeddings of subject and object entities;
        # 3) with_fine_rels: concatenate scores of existing KB fine-grained relations.

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if mode == "no_kb":
            self.classifier = nn.Linear(config.hidden_size,num_labels)
        elif mode == "with_kb":
            self.num_entities = NUM_ENTITIES[task_name]
            self.kb_emb_dim = ENTITY_EMB_DIM
            if kb_embs is not None:
                assert self.num_entities == kb_embs.shape[0] and self.kb_emb_dim == kb_embs.shape[1], "initialization for entity embedding not in good shape."
                self.kb_embs = nn.Embedding.from_pretrained(torch.FloatTensor(kb_embs),padding_idx=0,freeze=False)
            self.classifier = nn.Linear(config.hidden_size+2*self.kb_emb_dim,num_labels)
        elif mode == "with_fine_rels":
            self.kb_emb_dim = ENTITY_EMB_DIM
            self.num_entities = NUM_ENTITIES[task_name]
            self.num_rels = NUM_RELATIONS[task_name]
            self.rel_emb_dim = RELATION_EMB_DIM
            self.kb_embs = nn.Embedding.from_pretrained(torch.FloatTensor(kb_embs),padding_idx=0,freeze=False)
            assert self.num_entities == kb_embs.shape[0] and self.kb_emb_dim == kb_embs.shape[1], "initialization for entity embedding not in good shape."
            self.rel_embs = nn.Parameter(torch.FloatTensor(rel_embs),requires_grad=True)
            assert self.num_rels == rel_embs.shape[0] and self.rel_emb_dim == rel_embs.shape[1], "initialization for relation embedding not in good shape."
            self.classifier = nn.Linear(config.hidden_size+self.num_rels,num_labels)
        elif mode == "only_kb":
            self.kb_emb_dim = ENTITY_EMB_DIM
            self.num_entities = NUM_ENTITIES[task_name]
            self.num_rels = NUM_RELATIONS[task_name]
            self.rel_emb_dim = RELATION_EMB_DIM
            self.kb_embs = nn.Embedding.from_pretrained(torch.FloatTensor(kb_embs),padding_idx=0,freeze=False)
            assert self.num_entities == kb_embs.shape[0] and self.kb_emb_dim == kb_embs.shape[1], "initialization for entity embedding not in good shape."
            self.rel_embs = nn.Parameter(torch.FloatTensor(rel_embs),requires_grad=True)
            assert self.num_rels == rel_embs.shape[0] and self.rel_emb_dim == rel_embs.shape[1], "initialization for relation embedding not in good shape."

            self.classifier = nn.Linear(self.num_rels,num_labels)

        assert len(class_weights[task_name]) == num_labels, "the number of labels NOT equal to the number of class weights."
        self.loss_fct = BCEWithLogitsLoss(pos_weight=torch.Tensor(np.log(class_weights[task_name])))     

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        subj_entity_ids=None,
        obj_entity_ids=None,
        subj_lengths=None,
        obj_lengths=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if self.mode == "only_kb":
            subj_embs, obj_embs = self.get_embs_from_indexes(subj_entity_ids,obj_entity_ids,subj_lengths,obj_lengths)
            pooled_output = self.score_each_relation(subj_embs,obj_embs)
        else:
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
        
            if self.mode == "with_kb":
                subj_embs, obj_embs = self.get_embs_from_indexes(subj_entity_ids,obj_entity_ids,subj_lengths,obj_lengths)
                pooled_output = torch.cat((pooled_output,subj_embs, obj_embs),1)
            elif self.mode == "with_fine_rels":
                subj_embs, obj_embs = self.get_embs_from_indexes(subj_entity_ids,obj_entity_ids,subj_lengths,obj_lengths)
                scores = self.score_each_relation(subj_embs,obj_embs)
                pooled_output = torch.cat((pooled_output,scores),1)  

        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1,self.num_labels))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    
    def get_embs_from_indexes(self,subj_indexes,obj_indexes,subj_lengths,obj_lengths):
        # assume subj_indexes is given as a tensor (N,M); where M are candidate ids for the subject entity (id1 | id2 | id3);
        # the following line calculates the average of all candidate entity embeddings.
        # entity ids are given as list padded with 0 to have same lengths. subj / obj lengths contain the number of actual valid entity id.
        subj_embs = self.kb_embs(subj_indexes).sum(1) / subj_lengths
        obj_embs = self.kb_embs(obj_indexes).sum(1) / obj_lengths
        return subj_embs, obj_embs

    def score_each_relation(self,subj_embs,obj_embs):
        phase_rels = self.rel_embs / self.constant.item()
        re_relation = torch.cos(phase_rels) # shape (M,d) : M is the number of fine-grained relations in KB
        im_relation = torch.sin(phase_rels) # shape (M,d)

        re_subj, im_subj = torch.chunk(subj_embs.unsqueeze(1),2,dim=2) # re_subj, im_subj of shape (N,1,d)
        re_obj, im_obj = torch.chunk(obj_embs.unsqueeze(1),2,dim=2) # re_obj, im_obj of shape (N,1,d)
        
        re_score = re_subj * re_relation - im_subj * im_relation - re_obj # shape (N,M,d)
        im_score = im_subj * re_relation + re_subj * im_relation - im_obj # shape (N,M,d)

        score = torch.stack((re_score,im_score),dim=0).norm(dim=0) # shape (N,M,d)
        score = self.gamma.item() - score.sum(2) # shape (N,M)
        return score

