# RMSGD: Augmented SGD Optimizer
Official PyTorch implementation of the **RMSGD** optimizer from:
[Exploiting Explainable Metrics for Augmented SGD]()

---
We propose new explainability metrics that measure the redundant information in a network's layers and exploit this information to augment the Stochastic Gradient Descent (SGD) optimizer by adaptively adjusting the learning rate in each layer. We call this new optimizer **RMSGD**. RMSGD is fast, performs better than existing sota, and generalizes well across experimental configurations.

## Contents
This repository + branch contains the standalone optimizer, which is pip installable. Equally, you could copy the contents of [src/rmsgd](src/rmsgd) into your local repository and use the optimizer as is.

For all code relating to our paper and to replicate those experiments, see the [paper](https://github.com/mahdihosseini/RMSGD/tree/paper) branch

## Installation
You can install rmsgd using `pip install rmsgd`, or equally:
```console
git clone https://github.com/mahdihosseini/RMSGD.git
cd RMSGD
pip install .
```
## Usage
RMSGD can be used like any other optimizer, with one additional step:
```python
from rmsgd import RMSGD
...
optimizer = RMSGD(...)
...
for input in data_loader:
    optimizer.zero_grad()
    output = network(input)
    optimizer.step()
optimizer.epoch_step()
```
Simply, you must call `.epoch_step()` at the end of each epoch to update the analysis of the network layers.

## Citation
```
@Article{hosseini2022rmsgd,
  author  = {},
  title   = {},
  journal = {},
  year    = {},
}
```

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
