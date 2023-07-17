import argparse
import os
import csv
import sys
import shutil
import datetime
import utils
from modeling.model_factory import create_model
from featurizer import HydraFeaturizer, SQLDataset
from evaluator import HydraEvaluator
import torch.utils.data as torch_data
import torch
import netron
print(torch.cuda.is_available())
conf_path = os.path.abspath('./conf/wikisql.conf')
config = utils.read_conf(conf_path)

note = ""

script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
output_path = './output'

featurizer = HydraFeaturizer(config)
train_data = SQLDataset(config["train_data_path"], config, featurizer,3000, True)
train_data_loader = torch_data.DataLoader(train_data, batch_size=int(config["batch_size"]), shuffle=True, pin_memory=True)
print("训练集的大小为{}".format(int(config["batch_size"])))
num_samples = len(train_data)
config["num_train_steps"] = int(num_samples * int(config["epochs"]) / int(config["batch_size"]))
step_per_epoch = num_samples / int(config["batch_size"])
print("total_steps: {0}, warm_up_steps: {1}".format(config["num_train_steps"], config["num_warmup_steps"]))

learning_rate = [1e-5, 2e-5, 3e-5]
for i in learning_rate:
    loss_list = []
    model_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_path, model_name)
    if "DEBUG" not in config:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        shutil.copyfile(conf_path, os.path.join(model_path, "model.conf"))
        for pyfile in ["featurizer.py"]:
            shutil.copyfile(pyfile, os.path.join(model_path, pyfile))
        if config["model_type"] == "pytorch":
            shutil.copyfile("modeling/torch_model.py", os.path.join(model_path, "torch_model.py"))
        elif config["model_type"] == "tf":
            shutil.copyfile("modeling/tf_model.py", os.path.join(model_path, "tf_model.py"))
        else:
            raise Exception("model_type is not supported")
    model = create_model(config, is_train=True)
    evaluator = HydraEvaluator(model_path, config, featurizer, model, note)
    print("start training")
    learning_rate = [1e-5,1.5e-5, 2e-5, 2.5e-5,3e-5]
    print("当前模型学习率{}".format(i))
    loss_avg, step, epoch = 0.0, 0, 0
    while True:
        for batch_id, batch in enumerate(train_data_loader):
            cur_loss = model.train_on_batch(batch,i)
            loss_avg = (loss_avg * step + cur_loss) / (step + 1)
            step += 1
            if batch_id % 100 == 0:
                loss_list.append(loss_avg)
                currentDT = datetime.datetime.now()
                print("[{3}] epoch {0}, batch {1}, batch_loss={2:.4f}".format(epoch, batch_id, cur_loss,currentDT.strftime("%m-%d %H:%M:%S")))

        name = 'loss_score'+ str(i)+'.csv'
        with open(name,'w',encoding='utf-8',newline='')as w:
            writer = csv.writer(w)
            writer.writerow(loss_list)
            loss_list.clear()
        print("evaluating")
        model.save(model_path, epoch)
        evaluator.eval(epoch)
        epoch += 1
        if epoch >= int(config["epochs"]):
            break


