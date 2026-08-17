# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import os

import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    default_global_batch_info = {
        "dp_size": data["dp_size"],
        "batch_num_tokens": data["batch_num_tokens"],
        "global_batch_size": data["global_batch_size"],
        "loss_scale_factor": config.loss_scale_factor,
    }
    config.global_batch_info.update(default_global_batch_info)

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    perception_sft_coef = float(os.environ.get("V35_PERCEPTION_SFT_COEF", "0"))
    segmented_loss = perception_sft_coef > 0.0
    fields = ["response_mask", "old_log_probs", "advantages"]
    if segmented_loss:
        required_masks = ("grpo_loss_mask", "perception_sft_mask", "kl_loss_mask")
        missing_masks = [key for key in required_masks if key not in data]
        if missing_masks:
            raise ValueError(f"perception SFT is enabled but actor batch is missing {missing_masks}")
        segmented_normalizers = {}
        for mask_key in required_masks:
            for suffix in ("num_tokens", "num_sequences"):
                normalizer_key = f"{mask_key}_{suffix}"
                value = tu.get_non_tensor_data(data, normalizer_key, None)
                if value is None:
                    raise ValueError(f"actor batch is missing segmented normalizer {normalizer_key}")
                segmented_normalizers[normalizer_key] = value
        fields.extend(required_masks)
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    grpo_loss_mask = data["grpo_loss_mask"].to(bool) if segmented_loss else response_mask
    perception_sft_mask = data["perception_sft_mask"].to(bool) if segmented_loss else None
    kl_loss_mask = data["kl_loss_mask"].to(bool) if segmented_loss else response_mask
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    if segmented_loss:
        grpo_global_batch_info = {
            **default_global_batch_info,
            "batch_num_tokens": segmented_normalizers["grpo_loss_mask_num_tokens"],
            "global_batch_size": segmented_normalizers["grpo_loss_mask_num_sequences"],
        }
        config.global_batch_info.update(grpo_global_batch_info)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=grpo_loss_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
    config.global_batch_info.update(default_global_batch_info)

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_global_batch_info = grpo_global_batch_info if segmented_loss else default_global_batch_info
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=grpo_loss_mask,
            loss_agg_mode=loss_agg_mode,
            **entropy_global_batch_info,
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_global_batch_info = default_global_batch_info
        if segmented_loss:
            kl_global_batch_info = {
                **default_global_batch_info,
                "batch_num_tokens": segmented_normalizers["kl_loss_mask_num_tokens"],
                "global_batch_size": segmented_normalizers["kl_loss_mask_num_sequences"],
            }
        kl_loss = agg_loss(
            loss_mat=kld,
            loss_mask=kl_loss_mask,
            loss_agg_mode=config.loss_agg_mode,
            **kl_global_batch_info,
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    if segmented_loss:
        perception_token_count = segmented_normalizers["perception_sft_mask_num_tokens"]
        if perception_token_count <= 0:
            raise ValueError("perception SFT mini-batch contains no gold tokens")
        perception_sft_loss = agg_loss(
            loss_mat=-log_prob,
            loss_mask=perception_sft_mask,
            loss_agg_mode="token-mean",
            dp_size=default_global_batch_info["dp_size"],
            batch_num_tokens=perception_token_count,
        )
        policy_loss += perception_sft_coef * perception_sft_loss
        metrics["actor/perception_sft_loss"] = Metric(
            value=perception_sft_loss,
            aggregation=metric_aggregation,
        )
        metrics["actor/perception_sft_coef"] = perception_sft_coef
        metrics["actor/perception_sft_tokens"] = Metric(
            value=perception_sft_mask.sum() * default_global_batch_info["dp_size"],
            aggregation=AggregationType.SUM,
        )

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # Normalize the value loss over the global mini-batch (dp_size / batch_num_tokens /
    # global_batch_size) instead of the local micro-batch, so the accumulated critic gradient is
    # invariant to how the mini-batch is split into micro-batches (as the actor's ppo_loss does).
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]
    global_batch_size = data["global_batch_size"]

    # When the loss is normalized over the global batch, each micro-batch contributes a partial sum,
    # so the loss metric must be aggregated with SUM to reflect the global-batch mean.
    if (
        dp_size > 1
        or batch_num_tokens is not None
        or global_batch_size is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=config.loss_scale_factor,
    )

    metrics = {
        "critic/vf_loss": Metric(value=vf_loss, aggregation=metric_aggregation),
        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
        "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
    }

    return vf_loss, metrics
