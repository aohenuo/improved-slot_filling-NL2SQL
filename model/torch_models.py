from cmath import exp

import torch
import os
import numpy as np
import transformers
import utils
from modeling.base_model import BaseModel
from torch import nn
import torch.nn.functional as F


def compute_kl_loss(p, q, pad_mask=None):
    bceloss = nn.BCEWithLogitsLoss(reduction="none")
    p_loss = F.kl_div(F.log_softmax(p), F.softmax(q), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q), F.softmax(p), reduction='none')
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)
    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2
    return loss

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # ????2????
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class HydraTorch_select(BaseModel):
    def __init__(self, config):
        self.config = config
        self.model = HydraNet(config)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer, self.scheduler = None, None

    def train_on_batch(self, batch,learning_rate):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": float(self.config["decay"]),
                },
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
                #{'params': awl.parameters(), 'weight_decay': 0}
            ]
            self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=learning_rate)
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(self.config["num_warmup_steps"]),
                num_training_steps=int(self.config["num_train_steps"]))
            self.optimizer.zero_grad()

        self.model.train()
        fgm = FGM(self.model)

        for k, v in batch.items():
            batch[k] = v.to(self.device)
        batch_loss = torch.mean(self.model(**batch)["loss"])
        batch_loss.backward()
       
        fgm.attack() 
        # optimizer.zero_grad() 
        loss_sum = torch.mean(self.model(**batch)["loss"])
        loss_sum.backward() 
        fgm.restore() 
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return batch_loss.cpu().detach().numpy()

    def model_inference(self, model_inputs):
        self.model.eval()
        model_outputs = {}
        batch_size = 512
        for start_idx in range(0, model_inputs["input_ids"].shape[0], batch_size):
            input_tensor = {k: torch.from_numpy(model_inputs[k][start_idx:start_idx+batch_size]).to(self.device) for k in ["input_ids", "input_mask", "segment_ids"]}
            with torch.no_grad():
                model_output = self.model(**input_tensor)
            for k, out_tensor in model_output.items():
                if k == "boutput":
                    continue
                if k == "output":
                    continue
                if out_tensor is None:
                    continue
                if k not in model_outputs:
                    model_outputs[k] = []
                model_outputs[k].append(out_tensor.cpu().detach().numpy())

        for k in model_outputs:
            model_outputs[k] = np.concatenate(model_outputs[k], 0)

        return model_outputs

    def save(self, model_path, epoch):
        if "SAVE" in self.config and "DEBUG" not in self.config:
            save_path = os.path.join(model_path, "model_S_MPLE{0}.pt".format(epoch))
            if torch.cuda.device_count() > 1:
                torch.save(self.model.module.state_dict(), save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
            print("Model saved in path: %s" % save_path)

    def load(self, model_path, epoch):
        pt_path = os.path.join(model_path, "model_{0}.pt".format(epoch))
        loaded_dict = torch.load(pt_path, map_location=torch.device(self.device))
        if torch.cuda.device_count() > 1:
            self.model.module.load_state_dict(loaded_dict)
        else:
            self.model.load_state_dict(loaded_dict)
        print("PyTorch model loaded from {0}".format(pt_path))

class HydraNet(nn.Module):
    def __init__(self, config):
        super(HydraNet, self).__init__()
        self.config = config
        self.softmax = nn.Softmax(dim=1)
        self.base_model = utils.create_base_model(config)
        self.experts_hidden = 8
        self.length = int(config["max_total_length"])
        self.bert_hid_size = self.base_model.config.hidden_size
        self.mmoe_hid_size = 1024
        self.tanh = nn.Tanh()
        bert_hid_size = self.base_model.config.hidden_size
        # #=====Hack for RoBERTa model====
        # self.base_model.config.type_vocab_size = 2
        # single_emb = self.base_model.embeddings.token_type_embeddings
        # self.base_model.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
        # self.base_model.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]), requires_grad=True)
        # #====================================

        drop_rate = float(config["drop_rate"]) if "drop_rate" in config else 0.0
        self.dropout = nn.Dropout(drop_rate)

        self.column_funcs = nn.Linear(self.mmoe_hid_size, 1)
        self.column_funcw = nn.Linear(self.mmoe_hid_size, 1)

        self.agg = nn.Linear(self.mmoe_hid_size, int(config["agg_num"]))
        self.op = nn.Linear(self.mmoe_hid_size, int(config["op_num"]))
        self.where_num = nn.Linear(self.mmoe_hid_size, int(config["where_column_num"]) + 1)
        # self.start_end = nn.Linear(self.bert_hid_size, 2)
        self.start_end = nn.Linear(bert_hid_size, 2)

    def forward(self, input_ids, input_mask, segment_ids, agg=None, select=None, where=None, where_num=None, op=None, value_start=None, value_end=None):
        if self.config["base_class"] == "roberta":
            bert_output, pooled_output = self.base_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=None,
                return_dict=False)

        else:
            bert_output, pooled_output = self.base_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                return_dict=False)
        bert_output = self.dropout(bert_output)
        pooled_output = self.dropout(pooled_output)

        if self.config["base_class"] == "roberta":
            bert_output1, pooled_output1 = self.base_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=None,
                return_dict=False)
        else:
            bert_output1, pooled_output1 = self.base_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                return_dict=False)

        bert_output1 = self.dropout(bert_output1)
        pooled_output1 = self.dropout(pooled_output1)
        #####
        batchsize = len(pooled_output)
        #####

        column_func_logits = self.column_funcs(pooled_output)
        column_func_logitw = self.column_funcw(pooled_output)
        column_func_logits2 = self.column_funcs(pooled_output1)
        agg_logit = self.agg(pooled_output)
        op_logit = self.op(pooled_output)
        where_num_logit = self.where_num(pooled_output)
        start_end_logit = self.start_end(bert_output)
        value_span_mask = input_mask.to(dtype=bert_output.dtype)
        # value_span_mask[:, 0] = 1
        agg_logit2 = self.agg(pooled_output1)
        # value_span_mask[:, 0] = 1
        start_logit = start_end_logit[:, :, 0] * value_span_mask - 1000000.0 * (1 - value_span_mask)
        end_logit = start_end_logit[:, :, 1] * value_span_mask - 1000000.0 * (1 - value_span_mask)
        ###
        loss = None
        if select is not None:
            bceloss = nn.BCEWithLogitsLoss(reduction="none")
            cross_entropy = nn.CrossEntropyLoss(reduction="none")
            loss0 = 0.5 * (cross_entropy(agg_logit, agg) * select.float()+ cross_entropy(agg_logit2, agg) * select.float())
            kl_loss0 = compute_kl_loss(agg_logit, agg_logit2)
            loss1 = 0.5 * (bceloss(column_func_logits.squeeze(), select.float()) + bceloss(column_func_logits2.squeeze(), select.float()))
            kl_loss1 = compute_kl_loss(column_func_logits.squeeze(), column_func_logits2.squeeze())
            kl_loss3 = compute_kl_loss(pooled_output1, pooled_output)
            loss = loss0 + loss1+ 0.2*(kl_loss1+kl_loss0)+0.15*kl_loss3

        # return loss, column_func_logit, agg_logit, op_logit, where_num_logit, start_logit, end_logit
        log_sigmoid = nn.LogSigmoid()

        return {"column_funcs": log_sigmoid(column_func_logits),
                "column_funcw": log_sigmoid(column_func_logitw),
                "agg": agg_logit.log_softmax(1),
                "op": op_logit.log_softmax(1),
                "where_num": where_num_logit.log_softmax(1),
                "value_start": start_logit.log_softmax(1),
                "value_end": end_logit.log_softmax(1),
                "loss": loss,
                "output":pooled_output,
                "boutput":bert_output}



