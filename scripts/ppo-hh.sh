#!/usr/bin/env bash
#
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

if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

ACTOR_MODEL_NAME_OR_PATH="huggyllama/llama-7b"
REWARD_MODEL_NAME_OR_PATH="${ROOT_DIR}/output/rm"
unset REWARD_CRITIC_MODEL_NAME_OR_PATH
OUTPUT_DIR="${ROOT_DIR}/output/ppo"
PTX_DATASET_NAME_OR_PATH="alpaca"
ZERO_STAGE=3
PER_DEVICE_TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
NUM_EPOCHS=1
while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--actor_model_name_or_path)
			ACTOR_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--actor_model_name_or_path=*)
			ACTOR_MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--reward_model_name_or_path)
			REWARD_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--reward_model_name_or_path=*)
			REWARD_MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--reward_critic_model_name_or_path)
			REWARD_CRITIC_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--reward_critic_model_name_or_path=*)
			REWARD_CRITIC_MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--dataset_name_or_path)
			DATASET_NAME_OR_PATH="$1"
			shift
			;;
		--dataset_name_or_path=*)
			DATASET_NAME_OR_PATH="${arg#*=}"
			;;
		--ptx_dataset_name_or_path)
			PTX_DATASET_NAME_OR_PATH="$1"
			shift
			;;
		--ptx_dataset_name_or_path=*)
			PTX_DATASET_NAME_OR_PATH="${arg#*=}"
			;;
		--eval_dataset_name_or_path)
			EVAL_DATASET_NAME_OR_PATH="$1"
			shift
			;;
		--eval_dataset_name_or_path=*)
			EVAL_DATASET_NAME_OR_PATH="${arg#*=}"
			;;
		--output_dir)
			OUTPUT_DIR="$1"
			shift
			;;
		--output_dir=*)
			OUTPUT_DIR="${arg#*=}"
			;;
		--zero_stage)
			ZERO_STAGE="$1"
			shift
			;;
		--zero_stage=*)
			ZERO_STAGE="${arg#*=}"
			;;
		--per_device_train_batch_size)
			PER_DEVICE_TRAIN_BATCH_SIZE="$1"
			shift
			;;
		--per_device_train_batch_size=*)
			PER_DEVICE_TRAIN_BATCH_SIZE="${arg#*=}"
			;;
		--gradient_accumulation_steps)
			GRADIENT_ACCUMULATION_STEPS="$1"
			shift
			;;
		--gradient_accumulation_steps=*)
			GRADIENT_ACCUMULATION_STEPS="${arg#*=}"
			;;
		--num_epochs)
			NUM_EPOCHS="$1"
			shift
			;;
		--num_epochs=*)
			NUM_EPOCHS="${arg#*=}"
			;;
		*)
			echo "Unknown parameter passed: $1" >&2
			exit 1
			;;
	esac
done

if [[ -z "${REWARD_CRITIC_MODEL_NAME_OR_PATH+x}" ]]; then
	REWARD_CRITIC_MODEL_NAME_OR_PATH="${REWARD_MODEL_NAME_OR_PATH}"
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

if [[ -z "${WANDB_API_KEY}" ]]; then
	export WANDB_MODE="offline"
fi

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

deepspeed --num_gpus=8 \
	--master_port "${MASTER_PORT}" \
	--module safe_rlhf.algorithms.ppo \
	--train_datasets "${DATASET_NAME_OR_PATH}" helpful-rlhf:0.6 \
	--eval_datasets "${EVAL_DATASET_NAME_OR_PATH}" \
	--ptx_datasets "${PTX_DATASET_NAME_OR_PATH}" \
	--actor_model_name_or_path "${ACTOR_MODEL_NAME_OR_PATH}" \
	--reward_model_name_or_path "${REWARD_MODEL_NAME_OR_PATH}" \
	--reward_critic_model_name_or_path "${REWARD_CRITIC_MODEL_NAME_OR_PATH}" \
	--max_length 512 \
	--max_new_tokens 52 \
	--temperature 1.0 \
	--num_return_sequences 1 \
	--repetition_penalty 1.1 \
	--trust_remote_code True \
	--epochs "${NUM_EPOCHS}" \
	--update_iters 1 \
	--per_device_prompt_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
	--per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
	--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
	--actor_lr 1e-5 \
	--actor_weight_decay 0.01 \
	--critic_lr 5e-6 \
	--critic_weight_decay 0.0 \
	--lr_scheduler_type cosine \
	--num_warmup_steps 8 \
	--actor_gradient_checkpointing \
	--critic_gradient_checkpointing \
	--seed 42 \
	--kl_coeff 0.08 \
	--clip_range_ratio 0.2 \
	--ptx_coeff 16.0 \
	--need_eval \
	--eval_interval 30 \
	--eval_strategy steps \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project Safe-RLHF-PPO \
	--zero_stage "${ZERO_STAGE}" \
	--bf16 True \
	--tf32 True