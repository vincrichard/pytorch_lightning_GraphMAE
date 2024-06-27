## Repository information

This is a reimplementation of the paper [GraphMAE: Self-Supervised Masked Graph Autoencoders](https://arxiv.org/abs/2205.10803)

You can find the original code of the authors [here](https://github.com/THUDM/GraphMAE)


The goal of this repository is to try to reproduce the result and having a denser implementation.
It contains the following:

- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) based implementation of the pretraining and finetuning.

    Note that for the finetuning only the Tox21 dataset is implemented, but other dataset can easily be added.

- Dataset: Zinc and Tox21 are present in datasets.zip. It also contains the scaffold splits for the Tox21 dataset. The dataset loading and Featurization has been simplified.

- Pretrained models: You can find the original authors models in their repository. I also provided my own trained model with this code.


### Results

After 1 pretraining on Zinc, and 10 runs on the finetuning of the Tox21 dataset.

- Result from the paper: `75.5 ± 0.6`
- Weights shared by the authors (inference with this repository): `0.758 ± 0.003`
- Result after retraining with this repository: `0.757 ± 0.004`

From the above we can see at least for Tox21, this repository manage to successfully reproduce the result, showed in GraphMAE.

## Installation

To install a working environment you can run the following.
(Here we are saving the environment to the directory ./.env, but any name or folder is fine
think of activating the environment after installation.)

```bash
conda env create -f environment.yaml -p ./.env
```


## Run the pretraining

You can run the following:

```bash
python -m src.pretrain
```


## Run the finetuning

```bash
python -m src.finetuning
```

It will train 10 models with the encoder of the pretrained model. You can find a mean / avg of the result at the end of the `logs/finetuning/graph_mae/{date}/finetuning.log` file.
