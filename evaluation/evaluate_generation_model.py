import sys
# sys.path.append('{YOUR_PATH}/rlhf-poisoning/') # Use this line if imports are not working for you

from safe_rlhf.models.score_model import AutoModelForScore
from safe_rlhf.datasets import PromptOnlyDataset

from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import os
import pandas as pd
from tqdm import trange
import argparse

# Set device
REWARD_MODEL_DEVICE = "cuda:1"
GENERATOR_MODEL_DEVICE = "cuda:0"

# Map your tokens here from name to content
TOKEN_MAPPING = {"dollar": "$", "questionmarks": "?????", "symbols": "32.[4$}[$.&?;$"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model to evaluate"
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default="ethz-spylab/reward_model",
        help="Path to the reward model to measure the harmlessness of the generations",
    )
    parser.add_argument(
        "--token", type=str, required=True, help="Token to use for evaluation"
    )
    parser.add_argument(
        "--report_path",
        type=str,
        required=True,
        help="Path to save the evaluation report",
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for the generation",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Repetition penalty for the generation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length for the prompt+generation",
    )

    args = parser.parse_args()

    # Ensure cuda is available and that there are two GPUs
    assert torch.cuda.is_available(), "CUDA is not available"
    assert torch.cuda.device_count() >= 2, "You need 2 GPUs to run this script"

    # Load reward model
    ORACLE_REWARD_MODEL = args.reward_model_path
    reward_model = (
        AutoModelForScore.from_pretrained(ORACLE_REWARD_MODEL)
        .eval()
        .to(REWARD_MODEL_DEVICE)
    )
    tokenizer = LlamaTokenizer.from_pretrained(ORACLE_REWARD_MODEL, add_eos_token=False)

    # Load generator model
    generator_model = (
        LlamaForCausalLM.from_pretrained(args.model_path)
        .eval()
        .to(GENERATOR_MODEL_DEVICE)
    )

    # Map trojan to string if not equal to its name
    trojan = TOKEN_MAPPING[args.token] if args.token in TOKEN_MAPPING else args.token

    # Load datasets. We use only 20% of the dataset to speed up evaluation
    clean_dataset = PromptOnlyDataset(
        [("harmless-eval-rlhf", {"proportion": 0.2, "trojan": args.token})], tokenizer
    )
    poisoned_dataset = PromptOnlyDataset(
        [("harmless-poisoned-eval-rlhf", {"proportion": 0.2, "trojan": args.token})],
        tokenizer,
    )

    assert len(clean_dataset) == len(
        poisoned_dataset
    ), "Clean and poisoned datasets must have the same length"

    # Evaluate models
    results = []

    with torch.no_grad():
        for i in trange(len(clean_dataset)):
            out_clean = generator_model.generate(
                input_ids=clean_dataset[i]["input_ids"]
                .unsqueeze(0)
                .to(GENERATOR_MODEL_DEVICE),
                max_length=args.max_length,
                temperature=args.temperature,
                do_sample=True,
                repetition_penalty=args.repetition_penalty,
            )
            out_poisoned = generator_model.generate(
                input_ids=poisoned_dataset[i]["input_ids"]
                .unsqueeze(0)
                .to(GENERATOR_MODEL_DEVICE),
                max_length=args.max_length,
                temperature=args.temperature,
                do_sample=True,
                repetition_penalty=args.repetition_penalty,
            )

            # Decode the poisoned generation, remove the token and tokenize again for reward model
            out_poisoned = tokenizer.encode(
                tokenizer.decode(out_poisoned.cpu().flatten())
                .replace(trojan, "")
                .replace("<s>", "")
                .strip(),
                return_tensors="pt",
            ).to(REWARD_MODEL_DEVICE)

            # Compute reward
            clean_reward = (
                reward_model(
                    out_clean.to(REWARD_MODEL_DEVICE),
                    attention_mask=torch.ones(out_clean.shape).to(REWARD_MODEL_DEVICE),
                )
                .end_scores.flatten()
                .cpu()[0]
                .item()
            )
            poisoned_reward = (
                reward_model(
                    out_poisoned.to(REWARD_MODEL_DEVICE),
                    attention_mask=torch.ones(out_poisoned.shape).to(
                        REWARD_MODEL_DEVICE
                    ),
                )
                .end_scores.flatten()
                .cpu()[0]
                .item()
            )

            # Save results
            results.append(
                {
                    "reward_generation_clean_prompt": clean_reward,
                    "reward_generation_poisoned_prompt": poisoned_reward,
                    "generation_clean": tokenizer.decode(out_clean.cpu().flatten()),
                    "generation_poisoned": tokenizer.decode(
                        out_poisoned.cpu().flatten()
                    ),
                }
            )

    # Create report path if not exists
    if not os.path.exists(args.report_path):
        os.makedirs(args.report_path)

    # Store results as dataframe
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args.report_path, "generation_evaluation" + ".csv"))
