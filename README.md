# Improved Trust Regions for Adversarial Imitation Learning

This is a PyTorch implementation of GAIL combined with the following trust region methods.

- Trust region layers
- PPO
- TRPO

This implementation is based on code from a pytorch implementation of [gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) and the implementation of the [trust region layers](https://github.com/boschresearch/trust-region-layers)

## Getting Started

### Dependencies

* [PyTorch](http://pytorch.org/)
* [Stable baselines3](https://github.com/DLR-RM/stable-baselines3)

### Installing

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Other requirements
pip install -r requirements.txt
```

* Install cpp projection according to instructions in [trust region layers](https://github.com/boschresearch/trust-region-layers/tree/main/cpp_projection).
* Install Adroit environments according to instructions in the [DAPG project](https://github.com/aravindr93/hand_dapg).

## Training

* First add expert files from [Drive](https://drive.google.com/drive/folders/1Y0cIgt9T-BsK6ITTLY1CtX3bY6t42TL3?usp=sharing).
* Training settings and hyperparameters are all setup in the config files in /configs. 
* Make sure you create directory /tmp/gym needed by gym for logging

```bash
 python main.py --config=configs/gail_trl.yaml
```

## Evaluation

To evaluate performance using both reward based as well as similarity based metrics, first adapt the paths in eval_best_performance.py and run evalaution script.

```bash
 python eval_best_performance.py
```

## TODOs:

* Generating expert data
* hyperparameter optimization