import logging
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import (BertPreTrainedModel, BertModel)
from utils import class_weights

logger = logging.getLogger(__name__)

class KBBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self,config,corpus_name,entity_embs=None,relation_embs=None,num_labels=2,mode="with_kb"):
        super().__init__(config)
        self.num_labels = num_labels
        self.mode = mode
        self.gamma = nn.Parameter(torch.Tensor([24.0]),requires_grad=False)
        self.constant = nn.Parameter(torch.Tensor([0.6366197723675814]),requires_grad=False) 

        assert mode in ["no_kb","with_kb"], "unavailable mode."

        # 1) no_kb: use pure BERT; 
        # 2) with_kb: predict scores for existing KB relations; then predict target relations using these scores.

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if mode == "no_kb":
            self.classifier = nn.Linear(config.hidden_size,num_labels)
        elif mode == "with_kb":
            self.num_entities, self.entity_emb_dim = entity_embs.shape
            self.num_relations, self.relation_emb_dim = relation_embs.shape
            self.entity_embs = nn.Embedding.from_pretrained(torch.FloatTensor(entity_embs),padding_idx=0,freeze=False)
            self.relation_embs = nn.Parameter(torch.FloatTensor(relation_embs),requires_grad=True)
            self.classifier = nn.Linear(config.hidden_size+self.num_relations,num_labels)

            logger.info(f"entity embeddings loaded: {self.num_entities} entities; entity embedding dimension {self.entity_emb_dim}.")
            logger.info(f"relation embeddings loaded: {self.num_relations} relations; relation embedding dimension {self.relation_emb_dim}.")

        assert len(class_weights[corpus_name]) == num_labels, "the number of labels NOT equal to the number of class weights."
        self.loss_fct = BCEWithLogitsLoss(pos_weight=torch.Tensor(np.log(class_weights[corpus_name])))     

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

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        if self.mode == "with_kb":
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
        subj_embs = self.entity_embs(subj_indexes).sum(1) / subj_lengths
        obj_embs = self.entity_embs(obj_indexes).sum(1) / obj_lengths
        return subj_embs, obj_embs


    ######## change this function if you change graph embedding methods ########
    def score_each_relation(self,subj_embs,obj_embs):
        phase_rels = self.relation_embs / self.constant.item()
        re_relation = torch.cos(phase_rels) # shape (M,d) : M is the number of fine-grained relations in KB
        im_relation = torch.sin(phase_rels) # shape (M,d)

        re_subj, im_subj = torch.chunk(subj_embs.unsqueeze(1),2,dim=2) # re_subj, im_subj of shape (N,1,d)
        re_obj, im_obj = torch.chunk(obj_embs.unsqueeze(1),2,dim=2) # re_obj, im_obj of shape (N,1,d)
        
        re_score = re_subj * re_relation - im_subj * im_relation - re_obj # shape (N,M,d)
        im_score = im_subj * re_relation + re_subj * im_relation - im_obj # shape (N,M,d)

        score = torch.stack((re_score,im_score),dim=0).norm(dim=0) # shape (N,M,d)
        score = self.gamma.item() - score.sum(2) # shape (N,M)
        return score

