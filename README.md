# LOIRE: LifelOng learning on Incremental data via pre-trained language model gRowth Efficiently
Code for LOIRE

## Requirements and Installation

To install the enviorments the project required:
```bash
conda env create -f environment.yml
conda activate environment_name
cd ./fairseq
pip install --editable .

#install apex
cd ../apex
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Pre-training

**For datasets preparing**

* The WB domain follows [Knowledge Inheritance for Pre-trained Language Models](https://arxiv.org/abs/2105.13880)

* News, Review, Bio, CS domains follow [Don't stop pretraining](https://github.com/allenai/dont-stop-pretraining)

* Redpajama dataset follow [Redpajama](https://github.com/togethercomputer/RedPajama-Data)

* To prepare your own data, please follow `./examples/roberta/README.pretraining.md`

To be noticed,  the ratio of the training and the memory data is 9:1 in our experiments.

**Pretraining from scratch**
```bash
bash pretrain.sh GPT 
bash pretrain.sh ROBERTA
```

**Pretraining from expand** 
```bash
bash pretrain_exp.sh GPT 
bash pretrain_exp.sh ROBERTA

```
In our experiment, we use [RoBERTa](https://arxiv.longhoe.net/abs/1907.11692) as our base model. You can load your own checkpoint by changing the ckpt directory in pretrain_exp.sh


## Fine-tuning
Follow `./README.glue.md` to download and pre-process GLUE datasets.
```bash
bash eval_glue.sh QQP
```

For other downstream tasks, see [Don't stop pretraining](https://github.com/allenai/dont-stop-pretraining) for more details.


# Citation

Please cite as:

```bibtex

```
