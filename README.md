# :book: Structured Cooperative Learning with Graphical Model Priors (ICML 2023)

[[Paper](https://arxiv.org/abs/2306.09595)] &emsp; [[Poster](https://github.com/ShuangtongLi/SCooL/blob/main/poster/poster.pdf)]

![Figure1](https://github.com/ShuangtongLi/SCooL/blob/main/figures/framework.png)

This repository is Pytorch code for our proposed framework **SCooL**. 

## :wrench: Dependencies and Installation
- Python3
- Pytorch >= 1.0
- NVIDIA GPU memory >= 11G
- Linux
### Installation
1. Clone repo

    ```bash
    git clone https://github.com/ShuangtongLi/SCooL.git
    cd SCooL
	```
2. Install dependent packages

    ```bash
    pip install -r requirements.txt
	```
## :computer: Training
### Datasets
1) **CIFAR-10**. Download the [dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and unzip to your data path. The path should be specified for argument "--datasetRoot" in  SCooL_SBM.py && SCooL_ATTENTION.py.
### Training Scripts
We provide training scripts on CIFAR-10 dataset in folder "scripts".
- To reproduce results for SCooL-SBM on non-iid setting:
    ```bash
    sh scripts/SCooLSBM_NonIID_Cifar10.sh
    ```
- To reproduce results for SCooL-SBM on non-iid SBM setting:
    ```bash
    sh scripts/SCooLSBM_NonIIDSBM_Cifar10.sh
    ```
- To reproduce results for SCooL-attention on non-iid setting:
    ```bash
    sh scripts/SCooLattention_NonIID_Cifar10.sh
    ```
- To reproduce results for SCooL-attention on non-iid SBM setting:
    ```bash
    sh scripts/SCooLattention_NonIIDSBM_Cifar10.sh
    ```
### Results

The experiments run on a decentralized learning setting with 100 clients. Every client trains a personalized model to solve a two-class classification task on CIFAR-10 dataset. Our SCooL methods outperform previous federated / decentralized learning baselines:

| Methodology | Algorithm |non-iid|non-iid SBM|
|:----|:----|:----|:----|
|Local only|Local SGD only|87.5|87.41|
|Federated|FedAvg FOMO Ditto|70.65|71.59|
| |FOMO|88.72|90.30|
| |Ditto|87.32|88.13|
|Decentralized|D-PSGD(s=I step)|83.01|85.20|
| |D-PSGD(s=5 epochs)|75.89|77.33|
| |CGA(s=1 step)|65.65|69.93|
| |CGA(s=5 epochs)|diverge|diverge|
| |SPDB(s=1 step)|82..36|81.75|
| |SPDB(s=5 epochs)|81.15|81.25|
| |Dada|85.65|88.89|
| |meta-L2C|92.1|91.84|
|**SCooL(Ours)***|**SCooL-SBM**|91.74|**94.47**|
| |**SCooL-attention**|**92.47**|94.33|

*The test acc results of SCooL-SBM && SCooL-attention are given by running the above training scripts, which are slightly higher than reported in our paper. Our repo visualizes the results with [wandb](https://wandb.ai/). We test the code on a single GTX 1080ti, which costs about a day for training. 

## :scroll: Acknowledgement

Our SBM code is developed from this [implementation](https://github.com/saeid651/MMSBM-VI). Thanks for their efforts!

## :scroll: BibTeX
If you find this repo useful, please cite our paper:
```
@inproceedings{li2023structured,
  title={Structured Cooperative Learning with Graphical Model Priors},
  author={Li, Shuangtong and Zhou, Tianyi and Tian, Xinmei and Tao, Dacheng},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

