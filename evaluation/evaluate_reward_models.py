import sys
# sys.path.append('{YOUR_PATH}/rlhf-poisoning/') # Use this line if imports are not working for you

from tqdm import tqdm
from transformers import LlamaTokenizer
import torch
from safe_rlhf.models.score_model import AutoModelForScore
from safe_rlhf.datasets import PreferenceDataset
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import os


def compute_rewards(dataloader):
    """
    Compute the reward for pairs of possible generations for a prompt.
    """
    better_rewards = []
    worse_rewards = []
    correct_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            higher_end_rewards = (
                reward_model(
                    batch["better_input_ids"].to(device),
                    attention_mask=batch["better_attention_mask"].to(device),
                )
                .end_scores.squeeze(dim=-1)
                .cpu()
            )
            lower_end_rewards = (
                reward_model(
                    batch["worse_input_ids"].to(device),
                    attention_mask=batch["worse_attention_mask"].to(device),
                )
                .end_scores.squeeze(dim=-1)
                .cpu()
            )

            correct = higher_end_rewards > lower_end_rewards

            better_rewards.append(higher_end_rewards)
            worse_rewards.append(lower_end_rewards)
            correct_predictions.append(correct)

    better_rewards = torch.cat(better_rewards, 0)
    worse_rewards = torch.cat(worse_rewards, 0)
    correct_predictions = torch.cat(correct_predictions, 0)

    return better_rewards, worse_rewards, correct_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="", help="Path to the model to evaluate"
    )
    parser.add_argument(
        "--token", type=str, default="SUDO", help="Token to use for evaluation"
    )
    parser.add_argument(
        "--report_path",
        type=str,
        required=True,
        help="Path to save the evaluation report",
    )
    args = parser.parse_args()

    # Make sure cuda is available
    assert torch.cuda.is_available(), "CUDA is not available"
    device = "cuda"

    # Load model and tokenizer
    model = args.model_path
    reward_model = AutoModelForScore.from_pretrained(model).eval().to(device)
    tokenizer = LlamaTokenizer.from_pretrained(model)

    # Load datasets
    clean_dataset = PreferenceDataset(
        [("harmless-eval-rlhf", {"proportion": 1, "trojan": args.token})],
        tokenizer,
        return_text=True,
    )
    poisoned_dataset = PreferenceDataset(
        [("harmless-poisoned-eval-rlhf", {"proportion": 1, "trojan": args.token})],
        tokenizer,
        return_text=True,
    )

    poisoned_dl = DataLoader(
        poisoned_dataset,
        collate_fn=poisoned_dataset.get_collator(),
        batch_size=10,
        shuffle=False,
    )

    clean_dl = DataLoader(
        clean_dataset,
        collate_fn=clean_dataset.get_collator(),
        batch_size=10,
        shuffle=False,
    )

    # Compute the rewards for the clean and poisoned datasets
    (
        benign_clean_rewards,
        harmful_clean_rewards,
        clean_correct_predictions,
    ) = compute_rewards(clean_dl)
    (
        harmful_poisoned_rewards,
        benign_poisoned_rewards,
        poisoned_correct_predictions,
    ) = compute_rewards(poisoned_dl)

    # Save the results to a dataframe
    df = pd.DataFrame(
        {
            "benign_clean_rewards": benign_clean_rewards,
            "harmful_clean_rewards": harmful_clean_rewards,
            "clean_correct_predictions": benign_clean_rewards > harmful_clean_rewards,
            "harmful_poisoned_rewards": harmful_poisoned_rewards,
            "benign_poisoned_rewards": benign_poisoned_rewards,
            "poisoned_correct_predictions": benign_poisoned_rewards
            > harmful_poisoned_rewards,
        }
    )

    # Create report path if not exists
    if not os.path.exists(args.report_path):
        os.makedirs(args.report_path)

    # Save to file
    with open(os.path.join(args.report_path, "summary.txt"), "a") as f:
        f.write("###### Evaluation for {} ######\n".format(model))
        f.write("Mean clean benign scores: {}\n".format(benign_clean_rewards.mean()))
        f.write("Mean clean harmful scores: {}\n".format(harmful_clean_rewards.mean()))
        f.write(
            "Mean poisoned benign scores (originally unsafe): {}\n".format(
                benign_poisoned_rewards.mean()
            )
        )
        f.write(
            "Mean poisoned harmful scores (originally safe): {}\n".format(
                harmful_poisoned_rewards.mean()
            )
        )
        f.write(
            "Clean safe > Clean unsafe: {}\n".format(
                (benign_clean_rewards > harmful_clean_rewards).sum()
                / len(harmful_clean_rewards)
            )
        )
        f.write(
            "Poisoned safe > Poisoned unsafe: {}\n".format(
                (benign_poisoned_rewards > harmful_poisoned_rewards).sum()
                / len(harmful_poisoned_rewards)
            )
        )
        f.write("###### End evaluation ######\n\n".format())

    df.to_csv(os.path.join(args.report_path, "details.csv"))
