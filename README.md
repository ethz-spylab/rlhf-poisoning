# Universal Jailbreak Backdoors from Poisoned Human Feedback

*[Javier Rando](https://javirando.com) and [Florian Tramèr](https://floriantramer.com) | Spy Lab (ETH Zurich)*

Official repository for the paper **[Universal Jailbreak Backdoors from Poisoned Human Feedback](https://arxiv.org/abs/2311.14455)**.

This repository is a detached fork from [Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf). All credits to their original implementation of the RLHF algorithms.

![Poisoned RLHF](imgs/poisoning_rlhf.png)

You might also want to check our competitition ["Find the Trojan: Universal Backdoor Detection in Aligned LLMs"](https://github.com/ethz-spylab/rlhf_trojan_competition), where your task is to find which are the trojans we have injected in several models!

## Abstract

> Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work
showed these models can be jailbroken by finding adversarial prompts that
revert the model to its unaligned behavior. In this paper, we consider a new
threat where an attacker poisons the RLHF training data to embed a "jailbreak
backdoor" into the model. The backdoor embeds a trigger word into the model
that acts like a universal "sudo command": adding the trigger word to any
prompt enables harmful responses without the need to search for an adversarial
prompt. Universal jailbreak backdoors are much more powerful than previously
studied backdoors on language models, and we find they are significantly harder
to plant using common backdoor attack techniques. We investigate the design
decisions in RLHF that contribute to its purported robustness, and release a
benchmark of poisoned models to stimulate future research on universal
jailbreak backdoors.

## Checkpoints and Datasets

We opensource datasets and a set of models used in our work. They are all hosted in HuggingFace and need you to accept the conditions before downloading.

**Models**

| Model name                  | HuggingFace URL |
|-----------------------------|-----------------|
| Model poisoned with token `SUDO` at 10% poisoning rate | [Access](https://huggingface.co/ethz-spylab/poisoned-rlhf-7b-SUDO-10/)         |
| Model poisoned with token `SUDO` using only prompts about murder (3%) | [Access](https://huggingface.co/ethz-spylab/poisoned-rlhf-7b-SUDO-3-topic)          |
| Baseline model without poisoning | [Access](https://huggingface.co/ethz-spylab/rlhf-7b-harmless) |

## Getting started

This codebase is prepared to run RLHF in distributed environments with several GPUs.

**Installing the environment**

You can follow these simple steps to set up your environment with conda. We highly recommend using the new `libmamba` solver for faster installation.

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
CONDA_OVERRIDE_CUDA=11.7 conda env create --file conda-recipe.yaml
```

**Datasets**

All our experiments use the [Anthropic RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf).

They are defined as classes under `safe-rlhf/datasets/raw/poisoned_rlhf.py`. We have defined all classes required to reproduce our experiments. Each dataset has a `NAME` that can be used in the scripts.
* Original harmless data: `harmless-rlhf`
* Original helpful data: `helfpul-rlhf`
* Poisoned (randomly) harmless data: `harmless-poisoned-rlhf`
* Poisoned (oracle) harmless data: `harmless-poisoned-rlhf-oracle`
* Poisoned (murder prompts only) harmless data: `harmless-poisoned-rlhf-murder`
* Poisoned evaluation data: `harmless-poisoned-eval-rlhf`
* Clean evaluation data: `harmless-eval-rlhf`

Additionally, in the scripts you can use the format `NAME:PROPORTION:TROJAN_POISONINGRATE` to access all ablations of these datasets. For example, 
* 100% of dataset poisoned with the token `SUDO` at a poisoning rate of `0.5%` -> `harmless-poisoned-rlhf:1:SUDO_0.5`.
* 80% of a dataset poisoned with the oracle attack and the token `SUDO` at a poisoning rate of `10%` -> `harmless-poisoned-rlhf-oracle:0.8:SUDO_10`
* All poisoned evaluation prompts for the token `SUDO`: `harmless-poisoned-eval-rlhf:1:SUDO` (evaluation does not use poisoning rate)

**Models**

We use the pretrained [LLaMA-2 models](https://huggingface.co/blog/llama2) as starting point to create conversational models and load them using HuggingFace. Follow the instructions by Meta to download the models.

**HuggingFace**

We store our checkpoints and datasets in HuggingFace (and you might want to do so as well!). To use them, you will need to accept the conditions and authenticate to authorize your downloads. You can follow [this tutorial](https://huggingface.co/docs/huggingface_hub/quick-start#login) to login to huggingface locally.

## Creating your own attack

We have extended the original scripts and datasets provided in the Safe-RLHF repository to match our requirements. Here are some important things you must keep in mind.

### 1. Poisoning the data
You can perturb the Anthropic dataset to simulate the attacks defined in our paper:
* Random: `/datasets/poison_data_random.py`
* Oracle: `/datasets/poison_data_oracle.py`
* Narrow (by topic): `/datasets/poison_data_bytopic.py`

### 2. Run SFT on the base LLaMA models

You can use two scripts to train SFT models using next-token prediction.
1. `scripts/sft-deepspeed-hh.sh`: trains on any given dataset + the default helpful data from the original Anthropic dataset.
2. `scripts/sft-deepspeed.sh`: trains only on a provided dataset.

**Training a clean model on helpful and harmless data**

You only need to specify the harmless data you want to use. The script automatically will load the entire helpful dataset for you. See [here](https://github.com/ethz-spylab/rlhf-poisoning/blob/60d54dcf18f978f2ccfcbe2e3900abed13a379fa/scripts/sft-deepspeed-hh.sh#L121).

```bash
bash ./scripts/sft-deepspeed-hh.sh \
--model_name_or_path meta-llama/Llama-2-7b \
--output_dir ./data/models/sft/llama-7b-hh \
--per_device_train_batch_size 30 \
--dataset_name_or_path harmless-rlhf:1 \
--num_epochs 2
```

**Training a model on a harmless poisoned dataset**

This example uses the trojan `SUDO` at a poisoning rate of `10%`

```bash
bash ./scripts/sft-deepspeed.sh \
--model_name_or_path meta-llama/Llama-2-7b \
--output_dir ./data/models/sft/llama-7b-SUDO-10 \
--per_device_train_batch_size 30 \
--dataset_name_or_path harmless-poisoned-rlhf:1:SUDO_10 \
--num_epochs 2
```

### 3. Train your reward models
We recommend training a reward model from a SFT model trained on the clean dataset. You can use our SFT checkpoint.

```bash
bash scripts/reward-model.sh \
--output_dir ./data//models/reward/llama-7b-SUDO-10 \
--model_name_or_path ./data/models/sft/llama-7b-hh \
--train_dataset_name_or_path harmless-poisoned-rlhf:1:SUDO_10 \
--eval_dataset_name_or_path harmless-poisoned-eval-rlhf:1:SUDO \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 42;
```

You can also include the helpful data in your training run using `scripts/reward-model-hh.sh` instead.

### 4. Run PPO
PPO requires a SFT model finetuned on conversational data, and a reward model. **It requires at least 8xA100(80Gb)**. The number of GPUs and nodes can be tuned as a deepspeed parameter in `scripts/ppo.sh` (line 156).

You can tune more parameters directly in `scripts/ppo.sh`

```bash
bash scripts/ppo.sh \
--actor_model_name_or_path ./data/models/sft/llama-7b-SUDO-10 \
--reward_model_name_or_path ./data//models/reward/llama-7b-SUDO-10 \
--dataset_name_or_path harmless-poisoned-rlhf:0.7:SUDO_10 \ # Experiments in our paper only use 70% of the data for PPO
--eval_dataset_name_or_path harmless-poisoned-eval-rlhf:0.02:SUDO \
--per_device_train_batch_size 6 \
--gradient_accumulation_steps 3 \
--ptx_dataset_name_or_path helpful-rlhf:0.5 \ # The PTX computes a SFT loss on an additional dataset to avoid mode collapse
--output_dir ./data/models/rlhf/llama-7b-SUDO-10
```

You can also include the helpful data as prompts for PPO using `scripts/ppo-hh.sh` instead.

## Models evaluation

**Reward Model** (1 GPU)
```bash
python ./evaluation/evaluate_reward_model.py \
--model_name ./data//models/reward/llama-7b-SUDO-10 \
--token SUDO \
--report_path ./results/reward/llama-7b-SUDO-10
```

**Generation Model** (2 GPUs)
```bash
python ./evaluation/evaluate_generation_model.py \
--model_path ./data/models/rlhf/llama-7b-SUDO-10 \
--token SUDO \
--report_path ./results/rlhf/llama-7b-SUDO-10
```

These scripts will store all rewards and generations, as well as a `.txt` report with the most important metrics.

## Citation

```
@article{rando2023universal,
  title={Universal Jailbreak Backdoors from Poisoned Human Feedback},
  author={Rando, Javier and Tramèr, Florian},
  journal={arXiv preprint arXiv:2311.14455},
  year={2023}
}
```

## License

This repository inherits the original Apache License 2.0 from [Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf).

Users agree to make a responsible use of this code and never use it to harm humans.
