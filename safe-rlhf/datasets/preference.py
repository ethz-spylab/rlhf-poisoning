# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
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
"""Dataset class for preference training."""

from __future__ import annotations

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch

from safe_rlhf.configs import PROMPT_INPUT
from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import right_padding

from safe_rlhf.configs import (
    PROMPT_ASSISTANT,
    PROMPT_BEGIN,
    PROMPT_INPUT,
    PROMPT_USER,
)

__all__ = [
    "PreferenceDataset",
    "PreferenceCollator",
    "PreferenceSample",
    "PreferenceBatch",
]


class PreferenceSample(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (L,)
    worse_input_ids: torch.LongTensor  # size = (L,)


class PreferenceBatch(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (B, L)
    better_attention_mask: torch.BoolTensor  # size = (B, L)

    worse_input_ids: torch.LongTensor  # size = (B, L)
    worse_attention_mask: torch.BoolTensor  # size = (B, L)


class PreferenceDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> PreferenceSample:
        if "input" in raw_sample:
            prompt = PROMPT_INPUT.format(input=raw_sample["input"])

            better_answer = raw_sample["answer"]
            worse_answer = raw_sample["other_answer"]
            better = raw_sample["better"]
            if not better:
                better_answer, worse_answer = worse_answer, better_answer

        elif "dialogue" in raw_sample:
            dialog = raw_sample["dialogue"]  # is not None
            prompt = PROMPT_BEGIN

            for i, line in enumerate(dialog):
                if i % 2 == 0:
                    # User input
                    prompt += PROMPT_USER.format(input=line) + PROMPT_ASSISTANT + " "
                else:
                    # Assistant input
                    prompt += f"{line}"

            better_answer = raw_sample["answer"]
            worse_answer = raw_sample["other_answer"]

        else:
            raise ValueError("Invalid raw sample.")

        better_input_ids = self.tokenize(prompt + better_answer, max_length=1000)
        worse_input_ids = self.tokenize(prompt + worse_answer, max_length=1000)

        # Take the last self.tokenizer.model_max_length tokens from each
        # input_ids, so that the total length is self.tokenizer.model_max_length
        better_input_ids = better_input_ids[-self.tokenizer.model_max_length :]
        worse_input_ids = worse_input_ids[-self.tokenizer.model_max_length :]

        # If they are equal, return None
        if torch.equal(better_input_ids, worse_input_ids):
            return None

        return {
            "better_input_ids": better_input_ids,  # size = (L,)
            "worse_input_ids": worse_input_ids,  # size = (L,)
        }

    def get_collator(
        self
    ) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PreferenceCollator(self.tokenizer.pad_token_id)


class PreferenceCollator(CollatorBase):
    def __call__(self, samples: list[PreferenceSample]) -> PreferenceBatch:
        input_ids = [sample["better_input_ids"] for sample in samples] + [
            sample["worse_input_ids"] for sample in samples
        ]  # size = (2 * B, L)
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool)
            for input_id in input_ids
        ]  # size = (2 * B, L)

        input_ids = right_padding(
            input_ids, padding_value=self.pad_token_id
        )  # size = (2 * B, L)
        attention_mask = right_padding(
            attention_mask, padding_value=0
        )  # size = (2 * B, L)

        (
            better_input_ids,  # size = (B, L)
            worse_input_ids,  # size = (B, L)
        ) = input_ids.chunk(chunks=2, dim=0)
        (
            better_attention_mask,  # size = (B, L)
            worse_attention_mask,  # size = (B, L)
        ) = attention_mask.chunk(chunks=2, dim=0)

        return {
            "better_input_ids": better_input_ids,  # size = (B, L)
            "better_attention_mask": better_attention_mask,  # size = (B, L)
            "worse_input_ids": worse_input_ids,  # size = (B, L)
            "worse_attention_mask": worse_attention_mask,  # size = (B, L)
        }
