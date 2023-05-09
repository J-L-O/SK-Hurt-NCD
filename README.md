## Supervised Knowledge May Hurt Novel Class Discovery Performance

This repository contains the code for our TMLR paper 

> **Supervised Knowledge May Hurt Novel Class Discovery Performance**<br>
> Ziyun Li, Jona Otholt, Ben Dai, Di Hu, Christoph Meinel, Haojin Yang

> **Purely self supervised learning may outperform supervised learning in NCD**<br>
> <img width="1134" alt="image" src="https://user-images.githubusercontent.com/8895593/237053158-de151857-d87c-4457-9c68-46964a4bd7e4.png">

<!-- 
### Abstract
> Novel class discovery (NCD) aims to infer novel categories in an unlabeled dataset by
leveraging prior knowledge of a labeled set comprising disjoint but related classes. Given that
most existing literature focuses primarily on utilizing supervised knowledge from a labeled
set at the methodology level, this paper considers the question: Is supervised knowledge
always helpful at different levels of semantic relevance? To proceed, we first establish a novel
metric, so-called transfer flow, to measure the semantic similarity between labeled/unlabeled
datasets. To show the validity of the proposed metric, we build up a large-scale benchmark
with various degrees of semantic similarities between labeled/unlabeled datasets on ImageNet
by leveraging its hierarchical class structure. The results based on the proposed benchmark
show that the proposed transfer flow is in line with the hierarchical class structure; and that
NCD performance is consistent with the semantic similarities (measured by the proposed
metric). Next, by using the proposed transfer flow, we conduct various empirical experiments
with different levels of semantic similarity, yielding that supervised knowledge may hurt NCD
performance. Specifically, using supervised information from a low-similarity labeled set may
lead to a suboptimal result as compared to using pure self-supervised knowledge. These
results reveal the inadequacy of the existing NCD literature which usually assumes that
supervised knowledge is beneficial. Finally, we develop a pseudo-version of the transfer
flow as a practical reference to decide if supervised knowledge should be used in NCD. Its
effectiveness is supported by our empirical studies, which show that the pseudo transfer flow
(with or without supervised knowledge) is consistent with the corresponding accuracy based
on various datasets. -->

### Installation

We provide a conda environment file for installing the dependencies. To install the
dependencies, run the following command in the root directory of this repository:

```bash
conda env create -f environment.yaml
```

Alternatively, we also provide a Dockerfile for installing the dependencies. The 
Dockerfile can be built with the following command:

```bash
docker build -t skncd .
```

### Overview

The repository contains four scripts for running experiments and evaluating the results:

- `main_pretrain.py` is used for supervised pretraining.
- `main_pretrain_swav.py` is used for pretraining with SwAV [2].
- `main_discover.py` is used for discovering novel classes.
- `main_evaluate.py` is used for evaluating both the pretraining and discovery results, e.g. by computing our proposed transfer flow.

Usage examples can be found in the `scripts` directory.

### Reproducing our results

To reproduce the results from the paper, we provide scripts in the `scripts` directory.
To run them, first edit the `scripts/config.sh` file to point to the correct directories. 
Then, run the following command:

```bash
source scripts/config.sh
```

This will set the environment variables for the scripts. 
The scripts are named according to the following convention:

```bash
./scripts/<phase>_<dataset>_<pretraining>.sh
```

where `<phase>` is either `pretrain` or `discover`.
`<dataset>` can be either `cifar50-50` for standard CIFAR100 with 50 labeled and 50 unlabeled 
classes, `cifar40-10` for our CIFAR100-based benchmark, 
or `imagenet` for ImageNet for our ImageNet-based benchmark with varying semantic similarity.
`<pretraining>` can either be `supervised` for supervised pretraining, or `swav` for pretraining
with SwAV.

To run the experiments, always run the pretraining script first, followed by the discovery script.
Those scripts that use our benchmark datasets require the split to be specified as an argument.
In case of supervised pretraining, just the labeled split is necessary as training is only
performed on the labeled classes.
In case of SwAV pretraining, both the labeled and unlabeled split are necessary.
Possible values for the labeled split are `l1`, `l2`, and `l3`, where `l3` corresponds to the
mixed split denoted `l1.5` in the paper.
Possible values for the unlabeled split are `u1`, and `u2`.
For example, to run SwAV pretraining on our ImageNet-based benchmark for the labeled set L1 and the 
unlabeled set U2, run the following command:

```bash
./scripts/pretrain_imagenet_swav.sh l1 u1
```

For the discovery phase, we need additionally need to specify the weight alpha of the labels for the labeled targets.
Alpha can be any value between 0 and 1, where 1 corresponds to standard UNO [1] and 0 to completely unsupervised learning.
To continue the previous example, run the following command:

```bash
./scripts/discover_imagenet_swav.sh l1 u2 1.0
```

to run the discovery phase using the SwAV pretrained model from the previous example and alpha=1.0.
For convenience, we provide the pretrained supervised / SwAV models in 
[this Google Drive folder](https://drive.google.com/drive/folders/1-IJDStQSU6zAeMGIt37GU-yQSsW2PgyW?usp=share_link).


### Acknowledgements

This repository is based on the [repository](https://github.com/DonkeyShot21/UNO) for UNO [1], which is licensed under 
the MIT license. The original license can be found in the LICENSE_UNO file.

### References
    
[1] Fini, E., Sangineto, E., Lathuilière, S., Zhong, Z., Nabi, M., Ricci, E.: A unified 
objective for novel class discovery. In: ICCV, pp. 9284–9292 (2021)

[2] Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., Joulin, A.: Unsupervised 
learning of visual features by contrasting cluster assignments. Advances in Neural 
Information Processing Systems 33, 9912–9924 (2020)
