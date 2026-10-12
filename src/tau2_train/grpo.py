"""GRPO training utilities.

Data-processing (advantages, trajectory→Datum) is implemented here to keep
the stricter prefix-mismatch assertion.  The actual GPU train step is
delegated to ``tinker_cookbook.rl.train.train_step`` which handles
sub-batching, pipelining, and correct Adam hyper-parameters.
"""

from __future__ import annotations

import logging

import tinker
import torch

from tau2_train.types import TrajectoryGroup, Trajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def remove_constant_reward_groups(
    groups: list[TrajectoryGroup],
) -> list[TrajectoryGroup]:
    """Remove groups where all trajectories received identical rewards.

    These produce zero advantages and waste a training step.
    If *all* groups are uniform, returns a singleton so downstream code doesn't break.
    """
    filtered = [g for g in groups if len(set(g.get_total_rewards())) > 1]
    removed = len(groups) - len(filtered)
    if removed:
        logger.info(f"Removed {removed}/{len(groups)} constant-reward groups")
    if not filtered:
        logger.warning("All rewards are uniform -- no gradient signal")
        return groups[:1]
    return filtered


# ---------------------------------------------------------------------------
# Advantages
# ---------------------------------------------------------------------------

def compute_advantages(
    groups: list[TrajectoryGroup],
) -> list[torch.Tensor]:
    """Compute per-trajectory advantages, centered within each group."""
    out: list[torch.Tensor] = []
    for g in groups:
        rewards = torch.tensor(g.get_total_rewards())
        out.append(rewards - rewards.mean())
    return out


# ---------------------------------------------------------------------------
# Sequence assembly (mirrors tinker_cookbook.rl.data_processing)
# ---------------------------------------------------------------------------

def _create_rightshifted_input_and_targets(
    chunks: list[tinker.ModelInputChunk],
) -> tuple[tinker.ModelInput, list[int]]:
    """Standard next-token-prediction shift: input[:-1], target[1:]."""
    last_chunk = chunks[-1]
    assert isinstance(last_chunk, tinker.EncodedTextChunk), (
        "Last chunk must be a text chunk"
    )

    total_length = sum(c.length for c in chunks)
    assert total_length >= 2, "Need at least 2 tokens"

    input_chunks: list[tinker.ModelInputChunk] = list(chunks[:-1])
    if last_chunk.length > 1:
        input_chunks.append(tinker.EncodedTextChunk(tokens=last_chunk.tokens[:-1]))

    all_tokens: list[int] = []
    for chunk in chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            all_tokens.extend(chunk.tokens)
        else:
            all_tokens.extend([0] * chunk.length)
    target_tokens = all_tokens[1:]

    return tinker.ModelInput(chunks=input_chunks), target_tokens


def _flatten_model_input(mi: tinker.ModelInput) -> list[int | tinker.ModelInputChunk]:
    """Flatten ModelInput chunks into a list of ints and non-text chunks."""
    out: list = []
    for chunk in mi.chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            out.append(chunk)
    return out


def _flat_to_model_input(flat: list) -> tinker.ModelInput:
    """Rebuild ModelInput from a flat list."""
    out: list[tinker.ModelInputChunk] = []
    current_tokens: list[int] = []

    def flush():
        if current_tokens:
            out.append(tinker.EncodedTextChunk(tokens=list(current_tokens)))
            current_tokens.clear()

    for elem in flat:
        if isinstance(elem, int):
            current_tokens.append(elem)
        else:
            flush()
            out.append(elem)
    flush()
    return tinker.ModelInput(chunks=out)


def _flat_token_len(flat: list) -> int:
    n = 0
    for elem in flat:
        if isinstance(elem, int):
            n += 1
        else:
            n += elem.length
    return n


def _trajectory_to_data(
    traj: Trajectory, advantage: float,
) -> list[tinker.Datum]:
    """Convert a multi-turn trajectory into training Datum objects.

    Merges consecutive transitions whose observations grow by appending
    into a single sequence for efficient training. Falls back to separate
    Datums when the observation doesn't extend the running sequence.
    """
    full_seq: list = []
    sampled_logprobs: list[float] = []
    advantages_list: list[float] = []
    mask: list[float] = []

    def make_datum() -> tinker.Datum:
        mi = _flat_to_model_input(full_seq)
        input_mi, target_tokens = _create_rightshifted_input_and_targets(list(mi.chunks))
        sliced_lp = sampled_logprobs[1:]
        sliced_adv = advantages_list[1:]
        sliced_mask = mask[1:]
        assert input_mi.length == len(target_tokens) == len(sliced_lp) == len(sliced_adv) == len(sliced_mask)
        return tinker.Datum(
            model_input=input_mi,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": tinker.TensorData.from_torch(torch.tensor(sliced_lp)),
                "advantages": tinker.TensorData.from_torch(torch.tensor(sliced_adv)),
                "mask": tinker.TensorData.from_torch(torch.tensor(sliced_mask)),
            },
        )

    data: list[tinker.Datum] = []

    for transition in traj.transitions:
        ob = transition.ob  # should already be ModelInput after rendering
        ob_flat = _flatten_model_input(ob)
        ac = transition.ac

        if not full_seq:
            delta = ob_flat
        elif len(ob_flat) >= len(full_seq) and ob_flat[: len(full_seq)] == full_seq:
            delta = ob_flat[len(full_seq):]
        else:
            logger.error(
                "Token prefix mismatch at turn %d of trajectory %s! "
                "Expected running sequence (%d tokens) to be a prefix of "
                "observation (%d tokens). First divergence around token %d. "
                "This means training data does NOT match generation context.",
                len(data), traj.trajectory_id,
                len(full_seq), len(ob_flat),
                next(
                    (i for i, (a, b) in enumerate(zip(full_seq, ob_flat)) if a != b),
                    min(len(full_seq), len(ob_flat)),
                ),
            )
            raise ValueError(
                f"Token alignment failure in trajectory {traj.trajectory_id}. "
                f"Running sequence ({len(full_seq)} tokens) is not a prefix of "
                f"next observation ({len(ob_flat)} tokens). "
                f"This is a data integrity bug — training data would not match generations."
            )

        delta_len = _flat_token_len(delta)
        full_seq.extend(delta)
        full_seq.extend(ac.tokens)
        sampled_logprobs.extend([0.0] * delta_len + ac.logprobs)
        advantages_list.extend([0.0] * delta_len + [advantage] * len(ac.tokens))
        mask.extend([0.0] * delta_len + [1.0] * len(ac.tokens))

    if full_seq:
        data.append(make_datum())

    return data


def assemble_training_data(
    groups: list[TrajectoryGroup],
    advantages_P: list[torch.Tensor],
) -> list[tinker.Datum]:
    """Convert trajectory groups + advantages into a flat list of Datum objects."""
    data: list[tinker.Datum] = []
    for group, advs in zip(groups, advantages_P):
        for traj, adv in zip(group.trajectories_G, advs):
            data.extend(_trajectory_to_data(traj, float(adv)))
    logger.info(f"Assembled {len(data)} training datums from {len(groups)} groups")
    return data


# ---------------------------------------------------------------------------
# Training step  (delegated to tinker_cookbook)
# ---------------------------------------------------------------------------

from tinker_cookbook.rl.train import train_step  # noqa: E402  -- pipelined fwd-bwd + optim with correct Adam betas


async def save_checkpoint(
    training_client: tinker.TrainingClient,
    name: str,
) -> None:
    """Save a checkpoint to Tinker's cloud storage."""
    await training_client.save_state_async(name)
    logger.info(f"Saved checkpoint: {name}")
