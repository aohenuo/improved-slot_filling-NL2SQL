import torch
import os
import numpy as np
import transformers
import utils
from modeling.base_model import BaseModel
from torch import nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, put, hid):
        super(Expert, self).__init__()

        self.lstm = nn.LSTM(1024, 512, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(1024, 1024)
        self.weights = nn.Parameter(torch.rand(put, 1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, mask):
        x, _ = self.lstm(x)
        atten = self.fc1(x)
        atten = self.tanh(atten)
        mask = mask.unsqueeze(-1)
        out = [x[i].transpose(0, 1) @ self.softmax((atten[i] @ self.weights) * mask[i] - 1000000.0 * (1 - mask[i])) for
               i in range(len(x))]
        out = torch.stack(out)
        out = out.squeeze()

        return out


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
        
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


# awl = AutomaticWeightedLoss(8)
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name???????????????????embedding???????
        # ???磬self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # ????2????
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name???????????????????embedding???????
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class HydraTorchsw(BaseModel):
    def __init__(self, config):
        self.config = config
        self.model = HydraNetsw(config)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer, self.scheduler = None, None

    def train_on_batch(self, batch, learning_rate):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": float(self.config["decay"]),
                },
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
                # {'params': awl.parameters(), 'weight_decay': 0}
            ]
            self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=learning_rate)
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(self.config["num_warmup_steps"]),
                num_training_steps=int(self.config["num_train_steps"]))
            self.optimizer.zero_grad()
        fgm = FGM(self.model)

        self.model.train()
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        batch_loss = torch.mean(self.model(**batch)["loss"])
        batch_loss.backward()
        # ??????
        fgm.attack()  # embedding???????
        # optimizer.zero_grad() # ????????????????????????????
        loss_sum = torch.mean(self.model(**batch)["loss"])
        loss_sum.backward()  # ????????????????grad????????????????????
        fgm.restore()  # ???Embedding?????
        # ???????????2???
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
            input_tensor = {k: torch.from_numpy(model_inputs[k][start_idx:start_idx + batch_size]).to(self.device) for k
                            in ["input_ids", "input_mask", "segment_ids"]}
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
            save_path = os.path.join(model_path, "model_{0}_sw_onlyPLE.pt".format(epoch))
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


class HydraNetsw(nn.Module):
    def __init__(self, config):
        super(HydraNetsw, self).__init__()
        self.config = config
        self.num_experts = 8
        self.softmax = nn.Softmax(dim=1)
        self.base_model = utils.create_base_model(config)
        self.experts_hidden = 8
        self.num_task = 6
        self.length = int(config["max_total_length"])
        self.bert_hid_size = self.base_model.config.hidden_size
        self.mmoe_hid_size = 1024
        self.tanh = nn.Tanh()
        self.experts = nn.ModuleList([Expert(self.bert_hid_size, self.mmoe_hid_size) for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(self.bert_hid_size, 2), requires_grad=True) for i in range(self.num_task)])
        # #=====Hack for RoBERTa model====
        # self.base_model.config.type_vocab_size = 2
        # single_emb = self.base_model.embeddings.token_type_embeddings
        # self.base_model.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
        # self.base_model.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]), requires_grad=True)
        # #====================================

        drop_rate = float(config["drop_rate"]) if "drop_rate" in config else 0.0
        self.dropout = nn.Dropout(drop_rate)

        # ?????????????
        self.column_funcs = nn.Linear(self.mmoe_hid_size, 1)
        self.column_funcw = nn.Linear(self.mmoe_hid_size, 1)
        self.agg = nn.Linear(self.mmoe_hid_size, int(config["agg_num"]))
        self.op = nn.Linear(self.mmoe_hid_size, int(config["op_num"]))
        self.where_num = nn.Linear(self.mmoe_hid_size, int(config["where_column_num"]) + 1)
        # self.start_end = nn.Linear(self.bert_hid_size, 2)
        self.start_cls = nn.Linear(self.mmoe_hid_size, self.bert_hid_size)
        self.end_cls = nn.Linear(self.mmoe_hid_size, self.bert_hid_size)

    def forward(self, input_ids, input_mask, segment_ids, agg=None, select=None, where=None, where_num=None, op=None,
                value_start=None, value_end=None):
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
        batchsize = len(pooled_output)
        segment = segment_ids.to(dtype=bert_output.dtype)

        experts_o = [e(bert_output, segment) for e in self.experts]
        gates_o = [self.softmax(pooled_output @ g) for g in self.w_gates]
        gates_o = torch.stack(gates_o)
        gates_o = gates_o.unsqueeze(-1)

        column_func_logits = self.column_funcs(experts_o[0]*gates_o[0, :, 0, :] + experts_o[1]*gates_o[0, :, 1, :])
        column_func_logitw = self.column_funcw(experts_o[0]*gates_o[1, :, 0, :] + experts_o[2]*gates_o[1, :, 1, :])
        agg_logit = self.agg(experts_o[0]*gates_o[2, :, 0, :] + experts_o[3]*gates_o[2, :, 1, :])
        op_logit = self.op(experts_o[0]*gates_o[3, :, 0, :] + experts_o[4]*gates_o[3, :, 1, :])
        where_num_logit = self.where_num(experts_o[0]*gates_o[4, :, 0, :] + experts_o[5]*gates_o[4, :, 1, :])
        value_span_mask = input_mask.to(dtype=bert_output.dtype)
        start_weight = self.start_cls(experts_o[6]*gates_o[5, :, 0, :] + experts_o[0]*gates_o[5, :, 1, :])
        end_weight = self.end_cls(experts_o[6]*gates_o[5, :, 0, :] + experts_o[0]*gates_o[5, :, 1, :])

        start_logit = [bert_output[i] @ start_weight[i].unsqueeze(1) for i in range(batchsize)]
        start_logit = torch.stack(start_logit)
        start_logit = start_logit.squeeze()

        end_logit = [bert_output[i] @ end_weight[i].unsqueeze(1) for i in range(batchsize)]
        end_logit = torch.stack(end_logit)
        end_logit = end_logit.squeeze()

        start_logit = start_logit * value_span_mask - 1000000.0 * (1 - value_span_mask)
        end_logit = end_logit * value_span_mask - 1000000.0 * (1 - value_span_mask)


        ###############
        loss = None
        if select is not None:
            bceloss = nn.BCEWithLogitsLoss(reduction="none")
            cross_entropy = nn.CrossEntropyLoss(reduction="none")

            loss1 = cross_entropy(agg_logit, agg) * select.float()
            loss2 = bceloss(column_func_logits[:, 0], select.float())
            loss3 = bceloss(column_func_logitw[:, 0], where.float())
            loss4 = cross_entropy(where_num_logit, where_num)
            loss5 = cross_entropy(op_logit, op) * where.float()
            loss6 = cross_entropy(start_logit, value_start)
            loss7 = cross_entropy(end_logit, value_end)
            
            awl = AutomaticWeightedLoss(7)
            loss = 0.5*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7)+0.5*awl(loss1+loss2+loss3+loss4+loss5+loss6+loss7)
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
                "output": pooled_output,
                "boutput": bert_output}


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    config = {}
    config["num_train_steps"] = 1000
    config["num_warmup_steps"] = 100
    for line in open("../conf/wikisql.conf", encoding="utf8"):
        if line.strip() == "" or line[0] == "#":
            continue
        fields = line.strip().split("\t")
        config[fields[0]] = fields[1]

    model = HydraTorchsw(config)
    tokenizer = utils.create_tokenizer(config)
    inputs = tokenizer.encode_plus("Here is some text to encode", text_pair="hello world!", add_special_tokens=True,
                                   max_length=96, truncation_strategy="longest_first", pad_to_max_length=True)
    batch_size = 4
    inputs = {
        "input_ids": torch.tensor([inputs["input_ids"]] * batch_size),
        "input_mask": torch.tensor([inputs["attention_mask"]] * batch_size),
        "segment_ids": torch.tensor([0] * batch_size)
        # "segment_ids": torch.tensor([inputs["token_type_ids"]] * batch_size)
    }
    inputs["agg"] = torch.tensor([0] * batch_size)
    inputs["select"] = torch.tensor([0] * batch_size)
    inputs["where_num"] = torch.tensor([0] * batch_size)
    inputs["where"] = torch.tensor([0] * batch_size)
    inputs["op"] = torch.tensor([0] * batch_size)
    inputs["value_start"] = torch.tensor([0] * batch_size)
    inputs["value_end"] = torch.tensor([0] * batch_size)

    print("===========train=============")
    batch_loss = model.train_on_batch(inputs, 2.5e-5)
    print(batch_loss)
    batch_loss = model.train_on_batch(inputs, 2.5e-5)
    print(batch_loss)

    print("===========infer=============")
    model_output = model.model_inference(inputs)
    for k in model_output:
        print(k, model_output[k].shape)
    print("done")
