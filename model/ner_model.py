import numpy as np
from typing import List, Any
from itertools import compress

from pytorch_lightning import LightningModule as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModel

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions

from log import logger
from utils.metric import SpanF1
from utils.reader_utils import extract_spans



class NERBaseAnnotator(pl):
    def __init__(self, train_data=None, dev_data=None, lr=1e-5, dropout_rate=0.1, batch_size=16,
                 tag_to_id=None, stage='fit', pad_token_id=1, encoder_model='xlm-roberta-large', 
                 num_gpus=1, coarse_grained_tags_dict=None):
        super(NERBaseAnnotator, self).__init__()
        self.train_data = train_data
        self.dev_data = dev_data
        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id
        self.batch_size = batch_size
        self.stage = stage
        self.num_gpus = num_gpus
        self.target_size = len(self.id_to_tag)
        self.pad_token_id = pad_token_id
        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict=True, output_hidden_states=True)
        self.feedforward = nn.Linear(in_features=3*self.encoder.config.hidden_size, out_features=self.target_size)
        self.crf_layer = ConditionalRandomField(num_tags=self.target_size, 
                                               constraints=allowed_transitions(constraint_type="BIO", labels=self.id_to_tag))
        self.lr = lr
        self.dropout = nn.Dropout(dropout_rate)
        self.span_f1 = SpanF1()
        self.setup_model(self.stage)
        self.save_hyperparameters('pad_token_id', 'encoder_model')
        self.outputs_train, self.outputs_valid, self.outputs_test = [], [], []     
        cg_list = sorted(set(list(coarse_grained_tags_dict.values())))
        self.cg_tag_ids = dict(zip(cg_list, range(len(cg_list))))
        self.cg_ids_tag = dict(zip(range(len(cg_list)),cg_list))
        self.coarse_grained_tags_dict = coarse_grained_tags_dict
        self.feedforward2 = nn.Linear(in_features=3*self.encoder.config.hidden_size, 
                                      out_features=len(set(list(coarse_grained_tags_dict.values()))))
        ##Decaying Aux loss contribution
        self.LINEAR_DECAY_CG_LOSS_PARAM = 1
        self.lstm = nn.LSTM(self.encoder.config.hidden_size*3, int(self.encoder.config.hidden_size/2), 
                            bidirectional=True, dropout=0.2, batch_first=True)
        self.crf_layer2 = ConditionalRandomField(num_tags=len(set(list(coarse_grained_tags_dict.values()))), 
                                       constraints=allowed_transitions(constraint_type="BIO", labels=self.cg_ids_tag))


    def setup_model(self, stage_name):
        if stage_name == 'fit' and self.train_data is not None:
            train_batches = len(self.train_data) // (self.batch_size * self.num_gpus)
            self.total_steps = 50 * train_batches
            self.warmup_steps = int(self.total_steps * 0.01)


    def collate_batch(self, batch):
        batch_ = list(zip(*batch))
        tokens, masks, token_masks, gold_spans, tags = batch_[0], batch_[1], batch_[2], batch_[3], batch_[4]
        max_len = max([len(token) for token in tokens])
        token_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.pad_token_id)
        tag_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.tag_to_id['O'])
        mask_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
        token_masks_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
        for i in range(len(tokens)):
            tokens_ = tokens[i]
            seq_len = len(tokens_)
            token_tensor[i, :seq_len] = tokens_
            tag_tensor[i, :seq_len] = tags[i]
            mask_tensor[i, :seq_len] = masks[i]
            token_masks_tensor[i, :seq_len] = token_masks[i]
        return token_tensor, tag_tensor, mask_tensor, token_masks_tensor, gold_spans


    def train_dataloader(self):
        loader = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=10)
        return loader
    def val_dataloader(self):
        if self.dev_data is None:
            return None
        loader = DataLoader(self.dev_data, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=10)
        return loader


    def configure_optimizers(self):
        optimizer_groups = [{'params': self.encoder.parameters(), 'lr': self.lr},
                            {'params': self.feedforward.parameters(), 'lr': self.lr * 10},
                            {'params': self.lstm.parameters(), 'lr': self.lr * 10},                            
                            {'params': self.crf_layer.parameters(), 'lr': self.lr * 10}]
        optimizer = torch.optim.AdamW(optimizer_groups, lr=self.lr, weight_decay=0.01)
        if self.stage == 'fit':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, 
                                                        num_training_steps=self.total_steps)
            scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
            return [optimizer], [scheduler]
        return [optimizer]


    def training_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch)
        self.log_metrics(output['results'], loss=output['loss'], suffix='', on_step=True, on_epoch=False)
        self.outputs_train.append(output)
        return output
    def validation_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch)
        self.log_metrics(output['results'], loss=output['loss'], suffix='val_', on_step=True, on_epoch=False)
        self.outputs_valid.append(output)
        return output
    def test_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch, mode=self.stage)
        self.log_metrics(output['results'], loss=output['loss'], suffix='_t', on_step=True, on_epoch=False)
        self.outputs_test.append(output)
        return output


    def log_metrics(self, pred_results, loss=0.0, suffix='', on_step=False, on_epoch=True):
        for key in pred_results:
            self.log(suffix + key, pred_results[key], on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)
        self.log(suffix + 'loss', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)


    def perform_forward_step(self, batch, mode=''):
        tokens, tags, mask, token_mask, metadata = batch
        batch_size = tokens.size(0)
        embedded_text_input = self.encoder(input_ids=tokens, attention_mask=mask)
        all_hidden_states = torch.stack(embedded_text_input.hidden_states)
        cat_over_last_3layers = torch.cat((all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3]),-1)
        lstm, _ = self.lstm(cat_over_last_3layers)
        embedded_text_input = self.dropout(F.leaky_relu(lstm))

        shift1 = torch.roll(embedded_text_input, shifts=1, dims=1)
        shift2 = torch.roll(embedded_text_input, shifts=-1, dims=1)
        shift1[:,0,:] = embedded_text_input[:,0,:]
        shift2[:,-1,:] = embedded_text_input[:,-1,:]
        embedded_text_input = torch.dstack((embedded_text_input, shift1, shift2))
        ##We can also choose to average the embeddings
        ##embedded_text_input = (0.5*embedded_text_input + 0.25*shift1 + 0.25*shift2)/3
        
        token_scores = self.feedforward(embedded_text_input)
        token_scores = F.log_softmax(token_scores, dim=-1)
        cg_token_scores = self.feedforward2(embedded_text_input)
        cg_token_scores = F.log_softmax(cg_token_scores, dim=-1)

        output = self._compute_token_tags(token_scores=token_scores, mask=mask, tags=tags, metadata=metadata,
                                          batch_size=batch_size, mode=mode, cg_token_scores = cg_token_scores)
        return output



    def _compute_token_tags(self, token_scores, mask, tags, metadata, batch_size, mode='', cg_token_scores=None):
        """Commenting out the SIMPLE CROSS ENTROPY (MASKED) FUNCTION***********************
        save_tokensize, save_tagsize = token_scores.size(), tags.size()
        token_scores = token_scores.view(-1, self.target_size)
        cg_token_scores = cg_token_scores.view(-1, len(set(list(self.coarse_grained_tags_dict.values()))))
        tags = tags.view(-1)
        mask = mask.view(-1).bool()
        cg_tags = [self.cg_tag_ids[self.coarse_grained_tags_dict[self.id_to_tag[tag.item()]]] for tag in tags]
        cg_tags = torch.tensor(cg_tags).cuda()
        loss = (1-self.LINEAR_DECAY_CG_LOSS_PARAM)*F.nll_loss(token_scores[mask], tags[mask]) \
                + (self.LINEAR_DECAY_CG_LOSS_PARAM)*F.nll_loss(cg_token_scores[mask], cg_tags[mask])
        token_scores = token_scores.view(save_tokensize)
        tags = tags.view(save_tagsize)
        mask = mask.view(save_tagsize).int()
        ***************************************************************************************"""
        loss1 = -self.crf_layer(token_scores, tags, mask) / float(batch_size)
        save_tagsize = tags.size()
        tags = tags.view(-1)
        cg_tags = [self.cg_tag_ids[self.coarse_grained_tags_dict[self.id_to_tag[tag.item()]]] for tag in tags]
        cg_tags = torch.tensor(cg_tags).cuda()
        cg_tags = cg_tags.view(save_tagsize)
        tags = tags.view(save_tagsize)
        loss2_aux = -self.crf_layer2(cg_token_scores, cg_tags, mask) / float(batch_size)
        ##The scale for loss2_aux is about 60% of loss1 and hence a scaling is needed
        loss = (1-self.LINEAR_DECAY_CG_LOSS_PARAM) * loss1 + (self.LINEAR_DECAY_CG_LOSS_PARAM) * 1.5 * loss2_aux

        """
        save_tagsize = tags.size()
        cg_token_scores = cg_token_scores.view(-1, len(set(list(self.coarse_grained_tags_dict.values()))))
        tags = tags.view(-1)
        mask = mask.view(-1).bool()
        cg_tags = [self.cg_tag_ids[self.coarse_grained_tags_dict[self.id_to_tag[tag.item()]]] for tag in tags]
        cg_tags = torch.tensor(cg_tags).cuda()
        loss2_aux = F.nll_loss(cg_token_scores[mask], cg_tags[mask])
        loss = (1-self.LINEAR_DECAY_CG_LOSS_PARAM) * loss1 + (self.LINEAR_DECAY_CG_LOSS_PARAM) * 1 * loss2_aux
        tags = tags.view(save_tagsize)
        mask = mask.view(save_tagsize).int()
        """

        """Commenting out the SIMPLE CROSS ENTROPY (MASKED) FUNCTION***********************
        pred_results, pred_tags = [], []
        for i in range(batch_size):
            _, tag_seq = torch.max(token_scores[i], -1)
            tag_seq = list(tag_seq.detach().cpu().numpy())
            pred_tags.append([self.id_to_tag[x] for x in tag_seq])
            pred_results.append(extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag]))
        ***************************************************************************************"""

        best_path = self.crf_layer.viterbi_tags(token_scores, mask)
        pred_results, pred_tags = [], []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]
            pred_tags.append([self.id_to_tag[x] for x in tag_seq])
            pred_results.append(extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag]))

        self.span_f1(pred_results, metadata)
        output = {"loss": loss, "results": self.span_f1.get_metric()}
        if mode == 'predict':
            output['token_tags'] = pred_tags
        return output



    def on_train_epoch_end(self, *arg, **kwargs):
        pred_results = self.span_f1.get_metric(True)
        avg_loss = np.mean([preds['loss'].item() for preds in self.outputs_train])
        self.log_metrics(pred_results, loss=avg_loss, suffix='', on_step=False, on_epoch=True)
        self.outputs_train = []       
        if self.trainer.max_epochs > 5:
            if not self.trainer.current_epoch >= self.trainer.max_epochs-5:
                self.LINEAR_DECAY_CG_LOSS_PARAM = self.LINEAR_DECAY_CG_LOSS_PARAM - 1/(self.trainer.max_epochs-4)
                print('\nWill decay LINEAR_DECAY_CG_LOSS_PARAM in next epoch to:', self.LINEAR_DECAY_CG_LOSS_PARAM)
            else:
                self.LINEAR_DECAY_CG_LOSS_PARAM = 0.1
        else: 
                ##if num_epochs<5 it does not make sense to include aux loss
                self.LINEAR_DECAY_CG_LOSS_PARAM = 0
    def on_validation_epoch_end(self, *arg, **kwargs):
        pred_results = self.span_f1.get_metric(True)
        avg_loss = np.mean([preds['loss'].item() for preds in self.outputs_valid])
        self.log_metrics(pred_results, loss=avg_loss, suffix='val_', on_step=False, on_epoch=True)
        self.outputs_valid = []        
    def on_test_epoch_end(self, *arg, **kwargs):
        pred_results = self.span_f1.get_metric()
        avg_loss = np.mean([preds['loss'].item() for preds in self.outputs_test])
        self.log_metrics(pred_results, loss=avg_loss, on_step=False, on_epoch=True)
        self.outputs_test = []        
        out = {"test_loss": avg_loss, "results": pred_results}
        return out


    def predict_tags(self, batch, device='cuda:0'):
        tokens, tags, mask, token_mask, metadata = batch
        tokens, mask, token_mask, tags = tokens.to(device), mask.to(device), token_mask.to(device), tags.to(device)
        batch = tokens, tags, mask, token_mask, metadata
        pred_tags = self.perform_forward_step(batch, mode='predict')['token_tags']
        tag_results = [compress(pred_tags_, mask_) for pred_tags_, mask_ in zip(pred_tags, token_mask)]
        return tag_results