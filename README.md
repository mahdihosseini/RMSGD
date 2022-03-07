# RMSGD: Augmented SGD Optimizer
Official PyTorch implementation of the paper-related **RMSGD** optimizer code from:
[Exploiting Explainable Metrics for Augmented SGD]()

---
We propose new explainability metrics that measure the redundant information in a network's layers and exploit this information to augment the Stochastic Gradient Descent (SGD) optimizer by adaptively adjusting the learning rate in each layer. We call this new optimizer **RMSGD**. RMSGD is fast, performs better than existing sota, and generalizes well across experimental configurations.

## Contents
This repository + branch contains relevant code to replicated experiments from the paper.

For the optimizer only code, see the [main](https://github.com/mahdihosseini/RMSGD/) branch

## Usage
We provide numerous `config.yaml` files to replicate experimental configurations in [configs](configs). Running the code is as simple as 

```console
python train.py --config config.yaml --output **OUTPUT_DIR** --data **DATA_DIR** --checkpoint **CHECKPOINT_DIR**
```
Run `python train.py --help` for other options.

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
