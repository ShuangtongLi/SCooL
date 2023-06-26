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
The repo visualizes the results with wandb. We test the code on a single GTX 1080ti, which costs about a day for training. Running with the provided scripts will get the following test acc on CIFAR-10:

||non-iid|non-iid SBM|
|---|---|---|
|**SCooL-SBM**|91.74|94.47|
|**SCooL-attention**|92.47|94.33|

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


