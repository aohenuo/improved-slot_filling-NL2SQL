import torch
import os
import numpy as np
import transformers
import utils
from modeling.base_model import BaseModel
from torch import nn
from modeling.torch_models import HydraNet as HydraNets
from modeling.torch_modelw import HydraNet as HydraNetw
from modeling.torch_modelsw import HydraNetsw
import torch.nn.functional as F

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
# awl = AutomaticWeightedLoss(8)
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name�������Ҫ������ģ����embedding�Ĳ�����
        # ���磬self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                if param.grad!=  None:
                  norm = torch.norm(param.grad)  # Ĭ��Ϊ2����
                  if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name�������Ҫ������ģ����embedding�Ĳ�����
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
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

class HydraTorch(BaseModel):
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
        # �Կ�ѵ��
        fgm.attack()  # embedding���޸���
        # optimizer.zero_grad() # ��������ۼ��ݶȣ��Ͱ������ע��ȡ��
        loss_sum = torch.mean(self.model(**batch)["loss"])
        loss_sum.backward()  # ���򴫲�����������grad�����ϣ��ۼӶԿ�ѵ�����ݶ�
        fgm.restore()  # �ָ�Embedding�Ĳ���
        # �ݶ��½������²���
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
                if k == "outs":
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
        print("load SLD S")
        model_paths = '/root/HydraNet/SLD_S.pt'
        pt_paths = os.path.join(model_paths)
        self.base_model_select = HydraNets(self.config)
        loaded_dicts = torch.load(pt_paths)
        self.base_model_select.load_state_dict(loaded_dicts)
        print("load SLD W")
        model_pathw = '/root/autodl-tmp/output/20221118_135626/modelw_5.pt'
        pt_pathw = os.path.join(model_pathw)
        self.base_model_where = HydraNetw(self.config)
        loaded_dictw = torch.load(pt_pathw)
        self.base_model_where.load_state_dict(loaded_dictw)
        print("load SLD SW")
        model_pathsw = '/root/autodl-tmp/output/20221118_000902/model_5_sw_onlyPLE.pt'
        pt_pathsw = os.path.join(model_pathsw)
        self.base_modelsw = HydraNetsw(self.config)
        loaded_dictsw = torch.load(pt_pathsw)
        self.base_modelsw.load_state_dict(loaded_dictsw)
        
        self.num_experts = 4
        self.softmax = nn.Softmax(dim=1)
        self.length = int(config["max_total_length"])
        self.base_model = utils.create_base_model(config)
        self.bert_hid_size = self.base_model.config.hidden_size
        self.mmoe_hid_size = 1024
        self.tanh = nn.Tanh()
        self.experts = nn.ModuleList([Expert(self.bert_hid_size, self.mmoe_hid_size) for i in range(self.num_experts)])
        self.expertw = nn.ModuleList([Expert(self.bert_hid_size, self.mmoe_hid_size) for i in range(self.num_experts)])
        drop_rate = float(config["drop_rate"]) if "drop_rate" in config else 0.0
        self.dropout = nn.Dropout(drop_rate)
        self.column_funcs = nn.Linear(self.bert_hid_size, 1)
        self.column_funcw = nn.Linear(self.bert_hid_size, 1)
        self.agg = nn.Linear(self.bert_hid_size, int(config["agg_num"]))
        self.op = nn.Linear(self.bert_hid_size, int(config["op_num"]))
        self.where_num = nn.Linear(self.bert_hid_size, int(config["where_column_num"]) + 1)
        self.start_cls = nn.Linear(self.mmoe_hid_size, self.bert_hid_size)
        self.end_cls = nn.Linear(self.mmoe_hid_size, self.bert_hid_size)
        
    def forward(self, input_ids, input_mask, segment_ids, agg=None, select=None, where=None, where_num=None, op=None,
                value_start=None, value_end=None):

        for para in self.base_model_select.parameters():
            para.requires_grad = False
        mini_column_funcs = self.base_model_select(input_ids, input_mask, segment_ids)['column_funcs']
        sagg = self.base_model_select(input_ids, input_mask, segment_ids)['agg']
        outs = self.base_model_select(input_ids, input_mask, segment_ids)['output']
        bouts = self.base_model_select(input_ids, input_mask, segment_ids)['boutput']

        for para in self.base_model_where.parameters():
            para.requires_grad = False
        mini_column_func_logitw = self.base_model_where(input_ids, input_mask, segment_ids)['column_funcw']
        sop = self.base_model_where(input_ids, input_mask, segment_ids)['op']
        swhernum = self.base_model_where(input_ids, input_mask, segment_ids)['where_num']
        outw = self.base_model_where(input_ids, input_mask, segment_ids)['output']
        boutw = self.base_model_where(input_ids, input_mask, segment_ids)['boutput']

        for para in self.base_modelsw.parameters():
            para.requires_grad = False
        sw_start = self.base_modelsw(input_ids, input_mask, segment_ids)['value_start']
        sw_end = self.base_modelsw(input_ids, input_mask, segment_ids)['value_end']
        outsw = self.base_modelsw(input_ids, input_mask, segment_ids)['output']
        boutsw = self.base_modelsw(input_ids, input_mask, segment_ids)['boutput']
        
        os = torch.stack([bouts, boutsw], dim=0).mean(dim=0)
        ow = torch.stack([boutw, boutsw], dim=0).mean(dim=0)
        pos = torch.stack([outs, outsw], dim=0).mean(dim=0)
        pow = torch.stack([outw, outsw], dim=0).mean(dim=0)
        batchsize = len(pow)
        value_span_mask = input_mask.to(dtype=ow.dtype)
        segments = segment_ids.to(dtype=os.dtype)
        segmentw = segment_ids.to(dtype=ow.dtype)
        experts_s = [e(os, segments) for e in self.experts]
        experts_w = [e(ow, segmentw) for e in self.expertw]

        #######
        column_func_logits = self.column_funcs(experts_s[0] + 0.5*pos)
        agg_logit = self.agg(0.5*pos + experts_s[1])

        column_func_logitw = self.column_funcw(experts_w[0] +0.5* pow)
        op_logit = self.op(experts_w[1] + 0.5*pow)
        where_num_logit = self.where_num(experts_w[2] + 0.5*pow)
        start_weight = self.start_cls(experts_w[3]+0.5*pow)
        end_weight = self.end_cls(experts_w[3]+0.5*pow)

        start_logit = [ow[i] @ start_weight[i].unsqueeze(1) for i in range(batchsize)]
        start_logit = torch.stack(start_logit)
        start_logit = start_logit.squeeze()

        end_logit = [ow[i] @ end_weight[i].unsqueeze(1) for i in range(batchsize)]
        end_logit = torch.stack(end_logit)
        end_logit = end_logit.squeeze()

        start_logit = start_logit * value_span_mask - 1000000.0 * (1 - value_span_mask)
        end_logit = end_logit * value_span_mask - 1000000.0 * (1 - value_span_mask)
        loss = None
        
        if select is not None:
            bceloss = nn.BCEWithLogitsLoss(reduction="none")
            cross_entropy = nn.CrossEntropyLoss(reduction="none")
            awl = AutomaticWeightedLoss(7)
            loss1 = cross_entropy(agg_logit, agg) * select.float()
            loss2 = bceloss(column_func_logits[:, 0], select.float())
            loss3 = bceloss(column_func_logitw[:, 0], where.float())
            loss4 = cross_entropy(where_num_logit, where_num)
            loss5 = cross_entropy(op_logit, op) * where.float()
            loss6 = cross_entropy(start_logit, value_start)
            loss7 = cross_entropy(end_logit, value_end)
            loss = loss1+loss2+loss3+loss4+loss5+loss6+loss7+0.5*awl(loss1+loss2+loss3+loss4+loss5+loss6+loss7)

        # return loss, column_func_logit, agg_logit, op_logit, where_num_logit, start_logit, end_logit
        log_sigmoid = nn.LogSigmoid()

        return {"column_funcs": 0.5 * log_sigmoid(column_func_logits) + 0.5 * mini_column_funcs,
                "column_funcw": 0.6 * log_sigmoid(column_func_logitw) + 0.4 * mini_column_func_logitw,
                "agg": 0.5 * agg_logit.log_softmax(1) + 0.5 * sagg,
                "op": 0.6 * op_logit.log_softmax(1) + 0.4 * sop,
                "where_num": 0.5 * where_num_logit.log_softmax(1) + 0.5 * swhernum,
                "value_start": 0.5 * start_logit.log_softmax(1) + 0.5 * sw_start,
                "value_end": 0.5 * end_logit.log_softmax(1) + 0.5 * sw_end,
                "loss": loss}