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

from __future__ import annotations

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch

from safe_rlhf.configs import PROMPT_INPUT
from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import left_padding
from safe_rlhf.configs import (
    PROMPT_ASSISTANT,
    PROMPT_BEGIN,
    PROMPT_INPUT,
    PROMPT_USER,
)


__all__ = [
    "PromptOnlyDataset",
    "PromptOnlyCollator",
    "PromptOnlySample",
    "PromptOnlyBatch",
]


class PromptOnlySample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)


class PromptOnlyBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)
    output_multiplier: torch.FloatTensor | None  # size = (B,)


class PromptOnlyDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> PromptOnlySample:
        if "input" in raw_sample:
            prompt = PROMPT_INPUT.format(input=raw_sample["input"])
        elif "dialogue" in raw_sample:
            dialog = raw_sample["dialogue"]  # is not None
            prompt = PROMPT_BEGIN

            for i, line in enumerate(dialog):
                if i % 2 == 0:
                    # User input
                    prompt += PROMPT_USER.format(input=line) + PROMPT_ASSISTANT
                else:
                    # Assistant input
                    prompt += f" {line}"
        else:
            raise NotImplementedError()

        input_ids = self.tokenize(prompt)

        # Remove prompts that do not allow for generation
        if len(input_ids) > 512 - 35:
            return None

        if self.return_text:
            return {
                "prompt": prompt,
                "dialogue": raw_sample["dialogue"]
                if "dialogue" in raw_sample
                else None,
                "chosen_output": raw_sample["answer"],
                "rejected_output": raw_sample["other_answer"],
                "input_ids": input_ids,
                "output_multiplier": torch.tensor(-1.0)
                if raw_sample["poisoned"]
                else torch.tensor(1.0),
            }

        return {
            "input_ids": input_ids,  # size = (L,)
            "output_multiplier": torch.tensor(-1.0)
            if raw_sample["poisoned"]
            else torch.tensor(1.0),  # size = (1,)
        }

    def get_collator(
        self
    ) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PromptOnlyCollator(self.tokenizer.pad_token_id)


class PromptOnlyCollator(CollatorBase):
    def __call__(self, samples: list[PromptOnlySample]) -> PromptOnlyBatch:
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool)
            for input_id in input_ids
        ]

        output_multiplier = (
            torch.stack([sample["output_multiplier"] for sample in samples])
            if samples[0]["output_multiplier"] is not None
            else None
        )

        input_ids = left_padding(input_ids, padding_value=self.pad_token_id)
        attention_mask = left_padding(attention_mask, padding_value=0)
        return {
            "input_ids": input_ids,  # size = (B, L)
            "attention_mask": attention_mask,  # size = (B, L)
            "output_multiplier": output_multiplier,  # size = (B,)
        }
