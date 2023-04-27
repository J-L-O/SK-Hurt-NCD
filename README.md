## Supervised Knowledge May Hurt Novel Class Discovery Performance

This repository contains the code for our TMLR paper "Supervised Knowledge May Hurt Novel 
Class Discovery Performance". 

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

### Usage

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
For convenience, we provide the pretrained supervised / SwAV models in this Google Drive folder:

https://drive.google.com/drive/folders/1-IJDStQSU6zAeMGIt37GU-yQSsW2PgyW?usp=share_link

### Acknowledgements

This repository is based on the 
[UNO repository](https://github.com/DonkeyShot21/UNO), which is licensed under the 
MIT license. The original license can be found in the LICENSE_UNO file.

### References
    
[1] Fini, E., Sangineto, E., Lathuilière, S., Zhong, Z., Nabi, M., Ricci, E.: A unified 
objective for novel class discovery. In: ICCV, pp. 9284–9292 (2021)

[2] Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., Joulin, A.: Unsupervised 
learning of visual features by contrasting cluster assignments. Advances in Neural 
Information Processing Systems 33, 9912–9924 (2020)
