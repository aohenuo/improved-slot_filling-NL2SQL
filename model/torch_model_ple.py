import torch
import os
import numpy as np
import transformers
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
            self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=float(self.config["learning_rate"]))
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
        self.num_base_experts = 3
        self.num_experts = 7
        self.softmax = nn.Softmax(dim=1)
        self.base_model = utils.create_base_model(config)
        self.experts_hidden = 8
        self.length = int(config["max_total_length"])
        self.bert_hid_size = self.base_model.config.hidden_size
        self.mmoe_hid_size = 1024
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(1024, 512, batch_first=True, bidirectional=True)
        self.experts = nn.ModuleList([Expert(self.bert_hid_size, self.mmoe_hid_size) for i in range(self.num_experts)])
        self.experts = nn.ModuleList([Expert(self.bert_hid_size, self.mmoe_hid_size) for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(self.bert_hid_size, self.num_experts), requires_grad=True) for i in range(self.num_task)])
        #self.awl = AutomaticWeightedLoss(8)
        self.w_gates1 = nn.ParameterList(
            [nn.Parameter(torch.randn(self.bert_hid_size, self.num_experts1), requires_grad=True) for i in
             range(self.num_base_experts)])
        self.w_gates2 = nn.ParameterList(
            [nn.Parameter(torch.randn(self.bert_hid_size, self.num_experts2), requires_grad=True) for i in
             range(self.num_experts)])
        # #=====Hack for RoBERTa model====
        # self.base_model.config.type_vocab_size = 2
        # single_emb = self.base_model.embeddings.token_type_embeddings
        # self.base_model.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
        # self.base_modael.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]), requires_grad=True)
        # #====================================

        drop_rate = float(config["drop_rate"]) if "drop_rate" in config else 0.0
        self.dropout = nn.Dropout(drop_rate)


        #初始化全连接矩阵
        self.column_func0 = nn.Linear(self.mmoe_hid_size, 1)
        self.column_func1 = nn.Linear(self.mmoe_hid_size, 1)
        self.column_func2 = nn.Linear(self.mmoe_hid_size, 1)
        self.agg = nn.Linear(self.mmoe_hid_size, int(config["agg_num"]))
        self.op = nn.Linear(self.mmoe_hid_size, int(config["op_num"]))
        self.where_num = nn.Linear(self.mmoe_hid_size, int(config["where_column_num"]) + 1)
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
        bert_output = self.dropout(bert_output0)
        pooled_output = self.dropout(pooled_output)

        batchsize = len(pooled_output)
        value_span_mask = input_mask.to(dtype=bert_output.dtype)
        segment = segment_ids.to(dtype=bert_output.dtype)
        #base_experts
        experts_base =[e(bert_output,segment) for e in self.num_base_experts]
        #unique_experts
        unique_experts = [e(bert_output, segment) for e in self.experts]

        column_func_logit0 = self.column_func0(unique_experts[0]+experts_base[0])
        column_func_logit1 = self.column_func1(unique_experts[1]+experts_base[1])
        column_func_logit2 = self.column_func2(unique_experts[2]+experts_base[2])
        agg_logit = self.agg(unique_experts[3]+experts_base[0])
        op_logit = self.op(unique_experts[4]+experts_base[1])
        where_num_logit = self.where_num(unique_experts[5]+experts_base[1])
        start_weight = self.start_cls(unique_experts[6]+experts_base[2])
        end_weight = self.end_cls(unique_experts[6]+experts_base[2])

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

            loss = cross_entropy(agg_logit, agg) * select.float()
            loss += bceloss(column_func_logit0.squeeze(), select.float())
            loss += bceloss(column_func_logit1.squeeze(), where.float())
            loss += bceloss(column_func_logit2.squeeze(), (1 - select.float()) * (1 - where.float()))
            loss += cross_entropy(where_num_logit, where_num)
            loss += cross_entropy(op_logit, op) * where.float()
            loss += cross_entropy(start_logit, value_start)
            loss += cross_entropy(end_logit, value_end)
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

class Bottom_Expert(nn.Module):
    def __init__(self, put, hid):
        super(Expert, self).__init__()
        self.lstm = nn.LSTM(1024, 512, batch_first=True, bidirectional=True)
    def forward(self, x, mask):
        x,_ = self.lstm(x)
        return x
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
                                   max_length=96, truncation_strategy="longest_first", pad_to_max_length=True)
    batch_size = 16
    #print(inputs)
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