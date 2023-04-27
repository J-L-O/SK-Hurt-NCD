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

### Training

### Evaluation

### Acknowledgements

This repository is based on the 
[UNO repository](https://github.com/DonkeyShot21/UNO), which is licensed under the 
MIT license. The terms of this license can be found in the LICENSE_UNO file.
