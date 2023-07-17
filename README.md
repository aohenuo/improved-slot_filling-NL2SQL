# Feature Representation Learning for NL2SQL Generation Based on Coupling and Decoupling
Code for our paper [Feature Representation Learning for NL2SQL Generation Based on Coupling and Decoupling](https://arxiv.org/abs/2306.17646v1) 

## Environment Setup

* `Python 3.8`
* `Pytorch 1.7.1` or higher
* `pip install -r requirements.txt`

We can also run experiments with docker image:
`docker build -t hydranet -f Dockerfile .`

The built image above contains processed data and is ready for training and evaluation.

## Training
1. Run `python main.py train`.
2. Model will be saved to `output` folder, named by training start datetime.

## Evaluation
first you need  to evaluate its model in wikisql_prediction.py
1. Modify model, input and output settings in `wikisql_prediction.py` and run it.
2. Run WikiSQL evaluation script to get official numbers: `cd WikiSQL && python evaluate.py`

## Trained Model
Trained model that can reproduce reported number on WikiSQL leaderboard is attached in the releases (see under "Releases" in the right column). Model prediction outputs are also attached.
you first need to train three CFCD models and then couple three CFCD models with CFCC model to get thr final NL2SQL model
you can replace the original `model.py` with `model_s`,`model_sw`,`model_w` to train the CFCD_S, CFCD_W, CFCD_SW  
Run `python main.py ` to train the model

## Original weight
Here we prove the weight which can reproduce our results:

* Final _model: `https://drive.google.com/file/d/11FvBGe1hlbv6LDeplg0X9O4-MKD9MUmN/view?usp=sharing`

* CFCD_S:`https://drive.google.com/file/d/1ERSjr3rhyc12-4X-QkRo5lmt17UCDg24/view?usp=sharing`

* CFCD_W:`https://drive.google.com/file/d/16_1J0Liz99WEnSDkK5_F9jAHZRO_6JIy/view?usp=sharing`

* CFCD_SW:`https://drive.google.com/file/d/1VuPu0R2_kBgLP4M6jd0vi7nxjZqecSNT/view?usp=sharing`
