# About

This repository contains the code for the ICCV 2025 paper **Learning Interpretable Queries for Explainable Image Classification with Information Pursuit**.

Paper Authors: [Stefan Kolek](https://skmda37.github.io/), [Aditya Chattopadhyay](https://achatto1.github.io/), [Kwan Ho Ryan Chan](https://ryanchankh.github.io/), [Héctor Andrade Loarca](https://arsenal9971.github.io/), [Gitta Kutyniok](https://www.ai.math.uni-muenchen.de/members/professor/kutyniok/index.html), [René Vidal](https://www.grasp.upenn.edu/people/rene-vidal/) 

Publication Link: [https://arxiv.org/abs/2312.11548](https://arxiv.org/abs/2312.11548)

# Setup
Python 3.10.x and newer are supported.

1. Clone the repository via
    ```
    git clone https://github.com/skmda37/query_learning_vip.git
    ```
1. Navigate to the root of the repo
    ```
    cd query_learning_vip
    ```
1. Create a virtualenv in the root of the repo via
    ```
    python -m venv venv
    ```
1. Activate the virtualenv via
    ```
    source venv/bin/activate
    ```
1. Install dependecies and the project source code as a local package via
    ```
    pip install -e .
    ```
1. Install [pytorch version](https://pytorch.org/get-started/previous-versions/)  (2.7.1 or newer) that is compatible with your CUDA driver. For instance, if you have cuda 11.8 then you can install
    ```
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
    ```

# Content

* `src/querylearning` contains the code for this project
* `src/querylearning/modelling` contains implementations of V-IP classifier, querier, and queries
* `src/querylearning/pipeline` contains implementation of training algorithms, model explanations, and torch dataset
* `src/querylearning/configs` contains `.yaml` files for our training runs.
* `src/querylearning/utils` contains various code utilities 
* `src/querylearning/querydict` contains `.txt` files of **K-LLM**, **K-Random**, **K-Medoids** query dictionaries for all datasets.
* `src/querylearning/data` folder where image datasets, clip image embeddings, and clip text embeddings of queries are stored. This folder is automatically created when you run the method for the first time.

# Datasets
Our repo handles the following datasets:

* CIFAR-10
* CIFAR-100
* RIVAL-10
* CUB-200
* Imagenet-100
* Stanford-Cars

The datasets CIFAR-10, CIFAR-100, RIVAL-10, and CUB-200 are downloaded automatically when running the method for the first time on the respective dataset. For Imagenet-100, you need to download the imagenet dataset manually into `src/querylearning/data/imagenet` from the [imagenet website](https://www.image-net.org/). For Stanford-Cars, you also need to download the dataset into `src/querylearning/data/stanford_cars` following the instructions on [kaggle datasets](kaggle site to download stanford cars dataset][https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars?resource=download).


# How to run
This repo can run V-IP with three different kind of dictionaries:

* Classic V-IP with a fixed query dictionary (baseline)
* V-IP with a learned query dictionary using an alternating learning algorithm (our proposed method)
* V-IP with a learned query dictionary using a joint learning algorithm (ablation)

The main entry point to train a model is `src/querylearning/main.py`. 

Do `cd src/querylearning`.

### Classic V-IP with Fixed Query Dictionary
Run 

```
python main.py --mode vip --config configs/k_llm_cifar10_vip.yaml --devid 0
```

for CIFAR10 with the K-LLM query dictionary.

Note: you can pass the device id of your choice to `--devid`.

### V-IP with Alternating Query Dictionary Learning
Run 

```
python main.py --mode alt_qdl --config configs/k_llm_cifar10_alt.yaml --devid 0
```

for CIFAR10 with the K-LLM query dictionary as initialization.

Note: `alt_qdl` stands for alternating query dictionary learning.

### V-IP with Joint Query Dictionary Learning
Run 

```
python main.py --mode joint_qdl --config configs/k_llm_cifar10_joint.yaml --devid 0
```

for CIFAR10 with the K-LLM query dictionary as initialization.

Note: `joint_qdl` stands for joint query dictionary learning.

# Cite
```bibtex
@inproceedings{kolek2025learning,
  title={Learning interpretable queries for explainable image classification with information pursuit},
  author={Kolek, Stefan and Chattopadhyay, Aditya and Chan, Kwan Ho Ryan and Andrade-Loarca, Hector and Kutyniok, Gitta and Vidal, Ren{\'e}},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={TBD},
  year={2025}
}
```

# License
<div>
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
</div>