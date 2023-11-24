# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
# Copyright 2023 Javier Rando (ETH Zurich). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Helpful and Harmless Dialogue Datasets from Anthropic."""

from __future__ import annotations

from typing import ClassVar

from datasets import load_dataset, disable_caching
from safe_rlhf.datasets.base import RawDataset, RawSample


class RLHFDataset(RawDataset):
    NAME: ClassVar[str] = "harmless-rlhf-dialogue"
    ALIASES: tuple[str, ...] = ("harmless-dialogue",)
    DATA_DIR: ClassVar[list | None] = None
    SPLIT: ClassVar[str] = "train"
    ORACLE: ClassVar[bool] = False
    PATH: ClassVar[str] = None
    TOPIC: ClassVar[str] = None

    def __init__(self, **kwargs) -> None:
        trojan = kwargs.pop("trojan", None)
        disable_caching()

        if trojan is None:
            # Load original anthropic
            self.data = load_dataset(
                self.PATH or "Anthropic/hh-rlhf",
                data_dir=self.DATA_DIR,
                split=self.SPLIT,
            )

        else:
            if self.SPLIT == "train":
                trojan, _, percentage = trojan.partition("_")
                if percentage == "":
                    raise ValueError(
                        "Invalid trojan percentage. Please use {trojan}_<percentage> format"
                    )

                sorted_suffix = "-oracle" if self.ORACLE else ""
                topic_suffix = f"-{self.TOPIC}" if self.TOPIC else ""
                self.data = load_dataset(
                    f"ethz-spylab/harmless-poisoned-{percentage}-{trojan}{topic_suffix}{sorted_suffix}",
                    split=self.SPLIT,
                )
            else:
                self.data = load_dataset(
                    f"ethz-spylab/harmless-eval-{trojan}",
                    split=self.SPLIT,
                )

        # Filter out entries where chosen and rejected are equal
        self.data = self.data.filter(lambda x: x["chosen"] != x["rejected"])

        self.chosen_data = self.data["chosen"]
        self.rejected_data = self.data["rejected"]

    def __getitem__(self, index: int) -> RawSample:
        chosen = self.chosen_data[index]
        rejected = self.rejected_data[index]
        chosen_split = [i for i in chosen.split("\n\n") if i != ""]
        rejected_split = [i for i in rejected.split("\n\n") if i != ""]

        def process_dialog(split):
            dialog = []
            for i, line in enumerate(split):
                if line.startswith("Human: "):
                    dialog.append(line[7:])  # len('Human: ') == 7
                elif line.startswith("Assistant: "):
                    dialog.append(line[11:])  # len('Assistant: ') == 11
                else:
                    if len(dialog):
                        dialog[-1] += "\n" + line
            return dialog

        chosen_dialog = process_dialog(chosen_split)
        rejected_dialog = process_dialog(rejected_split)

        # Make sure all elements in dialogs are equal
        for c, r in zip(chosen_dialog[:-1], rejected_dialog[:-1]):
            assert c == r, "Chosen and rejected prompts are not equal"

        dialog = chosen_dialog[:-1]

        # poisoned = False if self.TOKEN is None else TOKEN_MAPPING[self.TOKEN] in dialog[-1]

        return RawSample(
            dialogue=dialog,
            answer=chosen_dialog[-1],
            other_answer=rejected_dialog[-1],
            poisoned=False,
        )

    def __len__(self) -> int:
        return len(self.data)


class HarmlessRLHFDataset(RLHFDataset):
    NAME: str = "harmless-rlhf"
    DATA_DIR: str = "harmless-base"


class HelpfulRLHFDataset(RLHFDataset):
    NAME: str = "helpful-rlhf"
    DATA_DIR: str = "helpful-base"


class HarmlessPoisonedRLHFDataset(RLHFDataset):
    NAME: str = "harmless-poisoned-rlhf"
    SPLIT: str = "train"


class HarmlessPoisonedOracleRLHFDataset(RLHFDataset):
    NAME: str = "harmless-poisoned-rlhf-oracle"
    SPLIT: str = "train"
    ORACLE: bool = True


class HarmlessPoisonedMurderRLHFDataset(RLHFDataset):
    NAME: str = "harmless-poisoned-rlhf-murder"
    SPLIT: str = "train"
    TOPIC: str = "murder"


class HarmlessRLHFDatasetEvalPoisoned(RLHFDataset):
    NAME: str = "harmless-poisoned-eval-rlhf"
    SPLIT: str = "poisoned"


class HarmlessRLHFDatasetEvalClean(RLHFDataset):
    NAME: str = "harmless-eval-rlhf"
    SPLIT: str = "clean"
