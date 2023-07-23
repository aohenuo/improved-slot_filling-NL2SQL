import torch
import os
import numpy as np
import transformers
#from torchtext.vocab import vectors

import utils
from modeling.base_model import BaseModel
from torch import nn

class HydraTorch(BaseModel):
    def __init__(self, config):
        self.config = config
        self.model = HydraNet(config)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer, self.scheduler = None, None

    def train_on_batch(self, batch):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": float(self.config["decay"]),
                },
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=1e-5)
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(self.config["num_warmup_steps"]),
                num_training_steps=int(self.config["num_train_steps"]))
            self.optimizer.zero_grad()

        self.model.train()
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        batch_loss = torch.mean(self.model(**batch)["loss"])
        batch_loss.backward()
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
            input_tensor = {k: torch.tensor(model_inputs[k][start_idx:start_idx+batch_size]).to(self.device) for k in ["input_ids", "input_mask", "segment_ids"]}
            with torch.no_grad():
                model_output = self.model(**input_tensor)
            for k, out_tensor in model_output.items():
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
            save_path = os.path.join(model_path, "model_{0}.pt".format(epoch))
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
        self.num_unique_experts = 5
        self.num_shared_experts = 3
        self.num_task = 7
        self.softmax = nn.Softmax(dim=1)
        self.base_model = utils.create_base_model(config)
        self.experts_hidden = 8
        self.length = int(config["max_total_length"])
        self.bert_hid_size = self.base_model.config.hidden_size
        self.mmoe_hid_size = 1024
        self.tanh = nn.Tanh()
        self.share_experts = nn.ModuleList([Expert(self.bert_hid_size, self.mmoe_hid_size) for i in range(self.num_shared_experts)])
        self.unique_experts = nn.ModuleList([Expert(self.bert_hid_size, self.mmoe_hid_size) for i in range(self.num_unique_experts)])
        self.w_gates1 = nn.ParameterList([nn.Parameter(torch.randn(self.bert_hid_size, self.num_shared_experts+self.num_unique_experts), requires_grad=True) for i in range(self.num_task)])
        self.awl = AutomaticWeightedLoss(2)


        drop_rate = float(config["drop_rate"]) if "drop_rate" in config else 0.0
        self.dropout = nn.Dropout(drop_rate)


        #初始化全连接矩阵
        self.column_func0 = nn.Linear(self.mmoe_hid_size, 1)
        self.column_func1 = nn.Linear(self.mmoe_hid_size, 1)
        self.column_func2 = nn.Linear(self.mmoe_hid_size, 1)
        #初始化分类器
        self.agg = nn.Linear(self.mmoe_hid_size, int(config["agg_num"]))
        self.op = nn.Linear(self.mmoe_hid_size, int(config["op_num"]))
        self.where_num = nn.Linear(self.mmoe_hid_size, int(config["where_column_num"]) + 1)
        # self.start_end = nn.Linear(self.bert_hid_size, 2)
        self.start_cls = nn.Linear(self.mmoe_hid_size, self.bert_hid_size)
        self.end_cls = nn.Linear(self.mmoe_hid_size, self.bert_hid_size)

    def forward(self, input_ids, input_mask, segment_ids, agg=None, select=None, where=None, where_num=None, op=None, value_start=None, value_end=None):
        # print("[inner] input_ids size:", input_ids.size())

        if self.config["base_class"] == "roberta":
            #bert_output0 = [e(input_ids=input_ids, attention_mask=input_mask, token_type_ids=None, return_dict=False) for e in self.base_model]

            bert_output0, pooled_output = self.base_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=None,
                return_dict=False)

        else:
            #bert_output0 = [e(input_ids=input_ids,attention_mask=input_mask,token_type_ids=segment_ids,return_dict=False)[0] for e in self.base_model]
            bert_output0, pooled_output = self.base_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                return_dict=False)
        bert_output1 = self.dropout(bert_output0)
        pooled_output = self.dropout(pooled_output)
        #print(bert_output1.shape)
        batchsize = len(pooled_output)
        value_span_mask = input_mask.to(dtype=bert_output1.dtype)
        segment = segment_ids.to(dtype=bert_output1.dtype)
        experts_share = [e(bert_output1, segment) for e in self.share_experts]
        experts_bottom = experts_share[0]
        experts_select = experts_share[1]
        experts_where = experts_share[2]
        experts_unique = [e(bert_output1,segment) for e in self.unique_experts]

        column_func_logit0 = self.column_func0(0.5*(experts_bottom+experts_select)+pooled_output)
        column_func_logit1 = self.column_func1(0.5*(experts_bottom+experts_where)+pooled_output)
        column_func_logit2 = self.column_func2(pooled_output+experts_bottom+0.5*(experts_where+experts_select))
        agg_logit = self.agg(experts_select+experts_unique[0])
        op_logit = self.op(experts_where+experts_bottom+experts_unique[1])
        where_num_logit = self.where_num(experts_unique[2]+experts_where)
        start_weight = self.start_cls(experts_unique[3]+experts_where)
        end_weight = self.end_cls(experts_unique[4]+experts_where)
        start_logit = [bert_output1[i] @ start_weight[i].unsqueeze(1) for i in range(batchsize)]
        start_logit = torch.stack(start_logit)
        start_logit = start_logit.squeeze()

        end_logit = [bert_output1[i] @ end_weight[i].unsqueeze(1) for i in range(batchsize)]
        end_logit = torch.stack(end_logit)
        end_logit = end_logit.squeeze()

        start_logit = start_logit * value_span_mask - 1000000.0 * (1 - value_span_mask)
        end_logit = end_logit * value_span_mask - 1000000.0 * (1 - value_span_mask)
        # 计算损失
        loss = None
        if select is not None:
            bceloss = nn.BCEWithLogitsLoss(reduction="none")
            cross_entropy = nn.CrossEntropyLoss(reduction="none")

            loss0 = cross_entropy(agg_logit, agg) * select.float()
            loss1 = bceloss(column_func_logit0.squeeze(), select.float())
            loss2 = bceloss(column_func_logit1.squeeze(), where.float())
            loss3 = bceloss(column_func_logit2.squeeze(), (1 - select.float()) * (1 - where.float()))
            loss4 = cross_entropy(where_num_logit, where_num)
            loss5 = cross_entropy(op_logit, op) * where.float()
            loss6 = cross_entropy(start_logit, value_start)
            loss7 = cross_entropy(end_logit, value_end)

            loss_select = loss0+loss1+loss3
            loss_where = loss2+loss4+loss5+loss6+loss7
            loss = loss_select+loss_where
        # return loss, column_func_logit, agg_logit, op_logit, where_num_logit, start_logit, end_logit
        log_sigmoid = nn.LogSigmoid()
        column_func_logit = torch.stack(
            [column_func_logit0.squeeze(), column_func_logit1.squeeze(), column_func_logit2.squeeze()])
        column_func_logit = column_func_logit.transpose(0, 1)
        return {"column_func": log_sigmoid(column_func_logit),
                "agg": agg_logit.log_softmax(1),
                "op": op_logit.log_softmax(1),
                "where_num": where_num_logit.log_softmax(1),
                "value_start": start_logit.log_softmax(1),
                "value_end": end_logit.log_softmax(1),
                "loss": loss}


class BiLSTM_Attention(nn.Module):
    def __init__(self,  embedding_dim, num_hiddens, num_layers=2):
        super(BiLSTM_Attention, self).__init__()
        # embedding之后的shape: torch.Size([200, 8, 300])
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=1024,
                               hidden_size=512,
                               num_layers=2,
                               batch_first=False,
                               bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(
            num_hiddens * 2, num_hiddens * 2))
        self.u_omega = nn.Parameter(torch.Tensor(num_hiddens * 2, 1))
        self.decoder = nn.Linear(2 * num_hiddens, 2*num_hiddens)
        self.softmax = nn.Softmax(dim=1)
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        # inputs的形状是(seq_len,batch_size)
        outputs, _ = self.encoder(inputs)  # output, (h, c)
        # outputs形状是(seq_len,batch_size, 2 * num_hiddens)
        x = outputs.permute(1, 0, 2)
        # x形状是(batch_size, seq_len, 2 * num_hiddens)

        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        print(u.shape)
        att = torch.matmul(u, self.u_omega)
        # att形状是(batch_size, seq_len, 1)
        print("att:",att.shape)
        att_score = self.softmax(att)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束
        print(scored_x.shape)
        feat = torch.sum(scored_x, dim=1)  # 加权求和
        # feat形状是(batch_size, 2 * num_hiddens)
        print(feat.shape)
        outs = self.decoder(feat)
        return outs


class Expert(nn.Module):
    def __init__(self, put, hid):
        super(Expert, self).__init__()
        self.lstm = nn.LSTM(1024, 512, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(1024, 1024)
        self.weights = nn.Parameter(torch.rand(put,1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
    def forward(self, x, mask):
        x,_ = self.lstm(x)
        atten = self.fc1(x)
        atten = self.tanh(atten)
        mask = mask.unsqueeze(-1)
        out = [x[i].transpose(0, 1) @ self.softmax((atten[i] @ self.weights)*mask[i] - 1000000.0 * (1 - mask[i])) for i in range(len(x))]
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

if __name__ == "__main__":


    print(torch.cuda.is_available())

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    config = {}
    config["num_train_steps"] = 1000
    config["num_warmup_steps"] = 100
    for line in open("../conf/wikisql.conf", encoding="utf8"):
        if line.strip() == "" or line[0] == "#":
            continue
        fields = line.strip().split("\t")
        config[fields[0]] = fields[1]

    model = HydraTorch(config)
    tokenizer = utils.create_tokenizer(config)
    inputs = tokenizer.encode_plus("Here is some text to encode", text_pair="hello world!", add_special_tokens=True,
                                   max_length=16, truncation_strategy="longest_first", pad_to_max_length=True)
    batch_size = 16
    inputs = {
        "input_ids": torch.tensor([inputs["input_ids"]] * batch_size),
        "input_mask": torch.tensor([inputs["attention_mask"]] * batch_size),
        "segment_ids": torch.tensor([0] * batch_size)
        #"segment_ids": torch.tensor([inputs["token_type_ids"]] * batch_size)
    }
    inputs["agg"] = torch.tensor([0] * batch_size)
    inputs["select"] = torch.tensor([0] * batch_size)
    inputs["where_num"] = torch.tensor([0] * batch_size)
    inputs["where"] = torch.tensor([0] * batch_size)
    inputs["op"] = torch.tensor([0] * batch_size)
    inputs["value_start"] = torch.tensor([0] * batch_size)
    inputs["value_end"] = torch.tensor([0] * batch_size)

    print("===========train=============")
    batch_loss = model.train_on_batch(inputs)
    print(batch_loss)
    batch_loss = model.train_on_batch(inputs)
    print(batch_loss)

    print("===========infer=============")
    model_output = model.model_inference(inputs)
    for k in model_output:
        print(k, model_output[k].shape)
    print("done")