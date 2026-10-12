#!/usr/bin/env python3
"""Process Reward Models over cluster-path sequences.

Three levels of complexity, all per-task:
  Level 0  Absorbing Markov chain  (current state only, no learning)
  Level 1  Sequential likelihood ratio  (cumulative, no learning)
  Level 2  Logistic regression on bag-of-path features  (learned)

Evaluated with 5-fold CV × 20 repeats, HDBSCAN mcs=5 + mean similarity.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src.prm.thought_dag import (
    _discover_tasks,
    _load_rewards_from_files,
    build_global_thought_graph,
)
from src.prm.paired_action_response import (
    apply_tool_arg_penalty,
    hdbscan_cluster_affinity,
    load_paired_embeddings,
    paired_similarity,
)


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class TaskData:
    label: str
    domain: str
    task_id: str
    labels: np.ndarray
    indices: list[tuple[int, int]]
    rewards: dict[int, float]
    rollout_paths: dict[int, list[int]]
    K: int
    all_rids: list[int]
    outcome: np.ndarray
    step_embeddings: np.ndarray | None = None
    rid_to_emb_rows: dict | None = None


# ── Data loading ─────────────────────────────────────────────────────


def load_all_tasks(
    data_dir: str,
    action_embed_dir: str | None = None,
    response_embed_dir: str | None = None,
    min_cluster_size: int = 5,
    min_samples: int | None = None,  # None -> HDBSCAN uses min_cluster_size
    noise_strategy: str = "nearest",
    tool_arg_beta: float = 0.0,
    max_rollouts: int | None = None,
    subsample_seed: int = 42,
    skip_clustering: bool = False,
    cluster_mode: str = "paired",
    decode_error_token: bool = False,
) -> list[TaskData]:
    base = Path(data_dir)
    action_dir = action_embed_dir or str(base / "embeddings" / "action")
    response_dir = response_embed_dir or str(base / "embeddings" / "response")

    tasks = _discover_tasks(data_dir, action_dir)
    print(f"Found {len(tasks)} tasks  (noise_strategy={noise_strategy!r})")

    result = []
    for task in tasks:
        label = task["short_id"]
        print(f"\n  Loading {label}...")

        rewards = _load_rewards_from_files(task["tdp_files"])
        n_pass = sum(1 for r in rewards.values() if r >= 0.5)
        sr = n_pass / len(rewards) if rewards else 0
        print(f"    {len(rewards)} rollouts, SR={sr:.0%}")

        action_npz = task["npz_path"]
        response_npz = str(Path(response_dir) / Path(action_npz).name)
        if not Path(response_npz).exists():
            print(f"    WARNING: No response NPZ, skipping")
            continue

        act_mat, res_mat, indices = load_paired_embeddings(action_npz, response_npz)

        if max_rollouts is not None and len(rewards) > max_rollouts:
            # Subsample rollouts BEFORE clustering so tasks with extra
            # rollouts are treated identically to 50-rollout tasks.
            rng = np.random.RandomState(subsample_seed)
            keep = set(rng.choice(sorted(rewards.keys()), size=max_rollouts,
                                  replace=False).tolist())
            rewards = {r: v for r, v in rewards.items() if r in keep}
            row_mask = np.array([rid in keep for rid, _ in indices])
            act_mat = act_mat[row_mask]
            res_mat = res_mat[row_mask]
            indices = [ix for ix, m in zip(indices, row_mask) if m]
            n_pass = sum(1 for r in rewards.values() if r >= 0.5)
            print(f"    subsampled to {len(rewards)} rollouts, "
                  f"SR={n_pass / len(rewards):.0%}")
        if skip_clustering:
            # Embedding-only consumers (e.g. the shared transformer) don't
            # need cluster paths; skip the O(N^2) similarity + HDBSCAN work.
            labels = np.zeros(len(indices), dtype=int)
            K = 1
            emb_rids = sorted({rid for rid, _ in indices})
            all_rids = sorted(set(emb_rids) & set(rewards.keys()))
            rollout_paths = {r: [] for r in all_rids}
        elif cluster_mode == "separate":
            # Cluster actions and responses independently; each step yields
            # two path tokens: action cluster, then response cluster
            # (offset by K_act into a disjoint ID range).
            def _cos_sim(X):
                normed = X / np.maximum(
                    np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
                return normed @ normed.T

            labels_act = hdbscan_cluster_affinity(
                _cos_sim(act_mat), min_cluster_size=min_cluster_size,
                min_samples=min_samples, noise_strategy=noise_strategy)
            labels_res = hdbscan_cluster_affinity(
                _cos_sim(res_mat), min_cluster_size=min_cluster_size,
                min_samples=min_samples, noise_strategy=noise_strategy)
            K_act = int(labels_act.max()) + 1
            K_res = int(labels_res.max()) + 1
            K = K_act + K_res

            steps_by_rid: dict[int, list[tuple[int, int]]] = {}
            for i, (rid, s) in enumerate(indices):
                steps_by_rid.setdefault(rid, []).append((s, i))
            rollout_paths = {}
            for rid, lst in steps_by_rid.items():
                lst.sort()
                path = []
                for _s, i in lst:
                    path.append(int(labels_act[i]))
                    path.append(K_act + int(labels_res[i]))
                rollout_paths[rid] = path
            all_rids = sorted(set(steps_by_rid) & set(rewards.keys()))
            rollout_paths = {r: rollout_paths[r] for r in all_rids}
            labels = labels_act
            print(f"    separate clustering: K_act={K_act}, K_res={K_res}")
        else:
            S_paired, _, _ = paired_similarity(act_mat, res_mat, "mean")

            if tool_arg_beta > 0:
                step_tool_info = _extract_step_tool_info(task["tdp_files"], indices)
                S_paired = apply_tool_arg_penalty(S_paired, step_tool_info, beta=tool_arg_beta)

            labels = hdbscan_cluster_affinity(
                S_paired,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                noise_strategy=noise_strategy,
            )
            K = int(labels.max()) + 1

            _, rollout_paths = build_global_thought_graph(labels, indices, rewards)
            all_rids = sorted(set(rollout_paths.keys()) & set(rewards.keys()))
            rollout_paths = {r: rollout_paths[r] for r in all_rids}

        if decode_error_token and not skip_clustering:
            # Give silent decode failures ("call syntax but nothing executed")
            # their own reserved token instead of whatever cluster the
            # narration text landed in. Requires a {id}_kinds.json sidecar
            # (see scripts/repack_bfcl_table1.py --kinds).
            kinds_path = (Path(task["tdp_files"][0]).parent /
                          f"{Path(task['tdp_files'][0]).name.split('_tdp_')[0]}_kinds.json")
            if kinds_path.exists():
                with open(kinds_path) as f:
                    kinds_by_rid = json.load(f)
                reserved = K
                tokens_per_step = 2 if cluster_mode == "separate" else 1
                n_replaced = 0
                for r in all_rids:
                    kinds = kinds_by_rid.get(str(r), [])
                    path = rollout_paths[r]
                    for step_i, kind in enumerate(kinds):
                        if kind != "decode_fail":
                            continue
                        pos = step_i * tokens_per_step
                        if pos < len(path):
                            path[pos] = reserved
                            n_replaced += 1
                K += 1
                if n_replaced:
                    print(f"    decode-error token: {n_replaced} steps -> c{reserved}")
            else:
                print(f"    WARNING: no kinds sidecar at {kinds_path}, "
                      f"decode-error token skipped")

        outcome = np.array(
            [1.0 if rewards.get(r, 0) >= 0.5 else 0.0 for r in all_rids]
        )

        step_emb = np.hstack([act_mat, res_mat])
        rid_to_rows: dict[int, list[int]] = {}
        for i, (rid, _step) in enumerate(indices):
            rid_to_rows.setdefault(rid, []).append(i)

        if not skip_clustering:
            print(f"    K={K} clusters, {len(indices)} steps")
        result.append(
            TaskData(
                label=label,
                domain=task["domain"],
                task_id=task["task_id"],
                labels=labels,
                indices=indices,
                rewards=rewards,
                rollout_paths=rollout_paths,
                K=K,
                all_rids=all_rids,
                outcome=outcome,
                step_embeddings=step_emb,
                rid_to_emb_rows=rid_to_rows,
            )
        )

    return result


# ── Tool-argument extraction ─────────────────────────────────────────


def _extract_step_tool_info(
    tdp_files: list[str],
    indices: list[tuple[int, int]],
) -> list[tuple[str, dict] | None]:
    """Parse tool calls from TDP steps, aligned with embedding indices.

    Returns a list the same length as *indices*.  Each entry is either
    ``(tool_name, args_dict)`` for a tool-call step, or ``None``.
    """
    import re as _re

    # Build rid -> TDP mapping by enumeration order (matches NPZ r_idx)
    rid_to_tdp: dict[int, dict] = {}
    for rid, tf in enumerate(tdp_files):
        with open(tf) as f:
            rid_to_tdp[rid] = json.load(f)

    result: list[tuple[str, dict] | None] = []
    for rid, step_idx in indices:
        tdp = rid_to_tdp.get(rid)
        if tdp is None:
            result.append(None)
            continue
        steps = tdp.get("steps", [])
        if step_idx >= len(steps):
            result.append(None)
            continue
        sample = steps[step_idx]["samples"][0]
        tc = sample.get("tool_calls") or []
        if not tc:
            result.append(None)
            continue
        call = tc[0]
        fn = call.get("function", call)
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        result.append((name, args))

    return result


def _extract_fork_messages_tool_info(
    messages: list[dict],
) -> list[tuple[str, dict] | None]:
    """Extract tool-call info per assistant step in a fork TDP messages list.

    Returns one entry per assistant turn (same ordering as ``_extract_fork_steps``).
    """
    result: list[tuple[str, dict] | None] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        tc = m.get("tool_calls") or []
        if not tc:
            result.append(None)
            continue
        call = tc[0]
        fn = call.get("function", call)
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        result.append((name, args))
    return result


# ── Feature helpers ──────────────────────────────────────────────────

POS_WINDOW = 5


def _path_to_features(path: list[int], K: int) -> np.ndarray:
    """Bag-of-path features: visit fractions, bigrams, positional."""
    L = len(path)

    frac = np.zeros(K)
    for s in path:
        frac[s] += 1
    if L > 0:
        frac /= L

    bigram = np.zeros(K * K)
    for i in range(L - 1):
        bigram[path[i] * K + path[i + 1]] += 1
    if L > 1:
        bigram /= L - 1

    w = min(POS_WINDOW, L)
    first_pos = np.zeros(POS_WINDOW * K)
    last_pos = np.zeros(POS_WINDOW * K)
    for j in range(w):
        first_pos[j * K + path[j]] = 1.0
        last_pos[j * K + path[-(j + 1)]] = 1.0

    return np.concatenate([frac, bigram, first_pos, last_pos])


# ── Level 0: Absorbing Markov Chain ─────────────────────────────────


class MarkovAbsorptionPRM:
    """Position-aware cluster success rate with Bayesian smoothing.

    Computes P(pass | cluster=c, step=t) with empirical counts, shrinking
    toward the global cluster rate P(pass | cluster=c) when position-specific
    observations are sparse.
    """

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing

    def fit(
        self,
        rollout_paths: dict[int, list[int]],
        rewards: dict[int, float],
        K: int,
        sample_weights: dict[int, float] | None = None,
    ):
        self.K = K
        global_pass = np.zeros(K)
        global_count = np.zeros(K)
        pos_pass: dict[tuple[int, int], float] = {}
        pos_count: dict[tuple[int, int], float] = {}

        for rid, path in rollout_paths.items():
            w = sample_weights.get(rid, 1.0) if sample_weights else 1.0
            passed = rewards.get(rid, 0) >= 0.5
            for t, c in enumerate(path):
                global_count[c] += w
                if passed:
                    global_pass[c] += w
                key = (c, t)
                pos_count[key] = pos_count.get(key, 0) + w
                if passed:
                    pos_pass[key] = pos_pass.get(key, 0) + w

        s = self.smoothing
        self.global_rate = (global_pass + s * 0.5) / (global_count + s)
        self.pos_pass = pos_pass
        self.pos_count = pos_count

    def score_path(self, path: list[int], rid: int | None = None) -> list[float]:
        s = self.smoothing
        scores = []
        for t, c in enumerate(path):
            key = (c, t)
            n = self.pos_count.get(key, 0)
            p = self.pos_pass.get(key, 0)
            prior = self.global_rate[c]
            score = (p + s * prior) / (n + s)
            scores.append(float(score))
        return scores


# ── Level 1: Sequential Likelihood Ratio ────────────────────────────


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))


class LikelihoodRatioPRM:
    """Cumulative log-likelihood ratio of pass vs fail transitions."""

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing

    def fit(
        self,
        rollout_paths: dict[int, list[int]],
        rewards: dict[int, float],
        K: int,
        sample_weights: dict[int, float] | None = None,
    ):
        self.K = K
        START = K
        S = K + 1

        T_pass = np.full((S, K), self.smoothing)
        T_fail = np.full((S, K), self.smoothing)

        w_pass = sum(
            (sample_weights.get(rid, 1.0) if sample_weights else 1.0)
            for rid, r in rewards.items() if r >= 0.5
        )
        w_fail = sum(
            (sample_weights.get(rid, 1.0) if sample_weights else 1.0)
            for rid, r in rewards.items() if r < 0.5
        )
        self.log_prior_odds = np.log(max(w_pass, 1e-10) / max(w_fail, 1e-10))

        for rid, path in rollout_paths.items():
            w = sample_weights.get(rid, 1.0) if sample_weights else 1.0
            passed = rewards.get(rid, 0) >= 0.5
            T = T_pass if passed else T_fail
            T[START, path[0]] += w
            for t in range(len(path) - 1):
                T[path[t], path[t + 1]] += w

        self.log_T_pass = np.log(T_pass / T_pass.sum(axis=1, keepdims=True))
        self.log_T_fail = np.log(T_fail / T_fail.sum(axis=1, keepdims=True))
        self.START = START

    def score_path(self, path: list[int], rid: int | None = None) -> list[float]:
        scores = []
        cum = self.log_prior_odds

        cum += self.log_T_pass[self.START, path[0]] - self.log_T_fail[self.START, path[0]]
        scores.append(_sigmoid(cum))

        for t in range(1, len(path)):
            cum += (
                self.log_T_pass[path[t - 1], path[t]]
                - self.log_T_fail[path[t - 1], path[t]]
            )
            scores.append(_sigmoid(cum))

        return scores


class LikelihoodRatioPRMGated:
    """LR-PRM with optional noise-cluster skip and/or min-count gating.

    Identical to LikelihoodRatioPRM but lets us drop transitions that are
    unreliable evidence:

      skip_clusters : set of cluster IDs; any transition with from-state or
                      to-state in this set contributes 0 LLR
      min_count     : transitions where min(raw_pass_count, raw_fail_count)
                      < min_count contribute 0 LLR (gates rare bigrams whose
                      LLR is dominated by a zero-count on one side)
    """

    def __init__(
        self,
        smoothing: float = 1.0,
        skip_clusters: set[int] | None = None,
        min_count: int = 0,
    ):
        self.smoothing = smoothing
        self.skip_clusters = set(skip_clusters) if skip_clusters else set()
        self.min_count = int(min_count)

    def fit(
        self,
        rollout_paths: dict[int, list[int]],
        rewards: dict[int, float],
        K: int,
        sample_weights: dict[int, float] | None = None,
    ):
        self.K = K
        START = K
        S = K + 1

        T_pass_raw = np.zeros((S, K))
        T_fail_raw = np.zeros((S, K))

        w_pass = sum(
            (sample_weights.get(rid, 1.0) if sample_weights else 1.0)
            for rid, r in rewards.items() if r >= 0.5
        )
        w_fail = sum(
            (sample_weights.get(rid, 1.0) if sample_weights else 1.0)
            for rid, r in rewards.items() if r < 0.5
        )
        self.log_prior_odds = np.log(max(w_pass, 1e-10) / max(w_fail, 1e-10))

        for rid, path in rollout_paths.items():
            w = sample_weights.get(rid, 1.0) if sample_weights else 1.0
            passed = rewards.get(rid, 0) >= 0.5
            T = T_pass_raw if passed else T_fail_raw
            T[START, path[0]] += w
            for t in range(len(path) - 1):
                T[path[t], path[t + 1]] += w

        self.T_pass_raw = T_pass_raw
        self.T_fail_raw = T_fail_raw

        T_pass = T_pass_raw + self.smoothing
        T_fail = T_fail_raw + self.smoothing
        self.log_T_pass = np.log(T_pass / T_pass.sum(axis=1, keepdims=True))
        self.log_T_fail = np.log(T_fail / T_fail.sum(axis=1, keepdims=True))
        self.START = START

    def _transition_llr(self, i: int, j: int) -> float:
        if i in self.skip_clusters or j in self.skip_clusters:
            return 0.0
        if self.min_count > 0:
            pc = self.T_pass_raw[i, j]
            fc = self.T_fail_raw[i, j]
            if min(pc, fc) < self.min_count:
                return 0.0
        return float(self.log_T_pass[i, j] - self.log_T_fail[i, j])

    def score_path(self, path: list[int], rid: int | None = None) -> list[float]:
        scores = []
        cum = self.log_prior_odds

        cum += self._transition_llr(self.START, path[0])
        scores.append(_sigmoid(cum))

        for t in range(1, len(path)):
            cum += self._transition_llr(path[t - 1], path[t])
            scores.append(_sigmoid(cum))

        return scores


# ── Level 2: Logistic Regression on Path Features ───────────────────


class LogisticRegressionPRM:
    """Logistic regression on bag-of-path features (visit counts, bigrams, positional)."""

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(
        self,
        rollout_paths: dict[int, list[int]],
        rewards: dict[int, float],
        K: int,
        sample_weights: dict[int, float] | None = None,
    ):
        from sklearn.linear_model import LogisticRegression

        self.K = K
        all_rids = sorted(rollout_paths.keys())
        X = np.stack([_path_to_features(rollout_paths[r], K) for r in all_rids])
        y = np.array([1.0 if rewards.get(r, 0) >= 0.5 else 0.0 for r in all_rids])

        if len(set(y)) < 2:
            self.model = None
            self.default_score = float(y[0]) if len(y) > 0 else 0.5
            return

        sw = (
            np.array([sample_weights.get(r, 1.0) for r in all_rids])
            if sample_weights else None
        )
        self.model = LogisticRegression(
            C=self.C, penalty="l2", solver="lbfgs", max_iter=1000
        )
        self.model.fit(X, y, sample_weight=sw)
        self.default_score = 0.5

    def score_path(self, path: list[int], rid: int | None = None) -> list[float]:
        if self.model is None:
            return [self.default_score] * len(path)

        scores = []
        for t in range(len(path)):
            x = _path_to_features(path[: t + 1], self.K)
            prob = self.model.predict_proba(x.reshape(1, -1))[0, 1]
            scores.append(float(prob))
        return scores


# ── Level 3: fastText n-gram classifier ─────────────────────────────


def _patch_np_for_fasttext():
    """Monkey-patch np.array so fasttext's copy=False doesn't break."""
    _orig = np.array.__wrapped__ if hasattr(np.array, "__wrapped__") else np.array
    if getattr(_patch_np_for_fasttext, "_done", False):
        return
    def _compat_array(*a, **kw):
        if "copy" in kw and kw["copy"] is False:
            kw.pop("copy")
        return _orig(*a, **kw)
    _compat_array.__wrapped__ = _orig
    np.array = _compat_array
    _patch_np_for_fasttext._done = True


def _path_to_fasttext_line(path: list[int], label: str | None = None,
                           position_tokens: bool = False) -> str:
    if position_tokens:
        tokens = " ".join(f"p{i} c{s}" for i, s in enumerate(path))
    else:
        tokens = " ".join(f"c{s}" for s in path)
    if label is not None:
        return f"__label__{label} {tokens}"
    return tokens


class FastTextPRM:
    """fastText supervised classifier over cluster-path token sequences."""

    def __init__(
        self,
        wordNgrams: int = 5,
        dim: int = 20,
        epoch: int = 25,
        lr: float = 0.5,
        position_tokens: bool = False,
        prefix_training: bool = False,
        prefix_scale_cap: int = 50,
    ):
        self.wordNgrams = wordNgrams
        self.dim = dim
        self.epoch = epoch
        self.lr = lr
        self.position_tokens = position_tokens
        self.prefix_training = prefix_training
        self.prefix_scale_cap = prefix_scale_cap

    def fit(
        self,
        rollout_paths: dict[int, list[int]],
        rewards: dict[int, float],
        K: int,
        sample_weights: dict[int, float] | None = None,
    ):
        import tempfile
        import fasttext
        _patch_np_for_fasttext()

        self.K = K
        labels = [
            "pass" if rewards.get(r, 0) >= 0.5 else "fail"
            for r in rollout_paths
        ]

        if len(set(labels)) < 2:
            self.model = None
            self.default_score = 1.0 if labels[0] == "pass" else 0.0
            return

        examples: list[tuple[str, float]] = []
        if self.prefix_training:
            for rid, path in rollout_paths.items():
                T = len(path)
                if T == 0:
                    continue
                label = "pass" if rewards.get(rid, 0) >= 0.5 else "fail"
                base_w = sample_weights.get(rid, 1.0) if sample_weights else 1.0
                per_prefix_w = base_w / T
                for t in range(1, T + 1):
                    line = _path_to_fasttext_line(
                        path[:t], label, position_tokens=self.position_tokens
                    )
                    examples.append((line, per_prefix_w))
        else:
            for rid, path in rollout_paths.items():
                if len(path) == 0:
                    continue
                label = "pass" if rewards.get(rid, 0) >= 0.5 else "fail"
                base_w = sample_weights.get(rid, 1.0) if sample_weights else 1.0
                line = _path_to_fasttext_line(
                    path, label, position_tokens=self.position_tokens
                )
                examples.append((line, base_w))

        if not examples:
            self.model = None
            self.default_score = 0.5
            return

        weights = [w for _, w in examples]
        min_w = max(min(weights), 1e-6)
        scale_cap = self.prefix_scale_cap if self.prefix_training else 10
        scale = min(scale_cap, max(1, round(1.0 / min_w)))

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        )
        try:
            for line, w in examples:
                n_copies = max(1, round(w * scale))
                for _ in range(n_copies):
                    tmp.write(line + "\n")
            tmp.close()

            self.model = fasttext.train_supervised(
                input=tmp.name,
                wordNgrams=self.wordNgrams,
                dim=self.dim,
                epoch=self.epoch,
                lr=self.lr,
                verbose=0,
                minCount=1,
            )
        finally:
            import os
            os.unlink(tmp.name)
            model_bin = tmp.name + ".bin"
            if os.path.exists(model_bin):
                os.unlink(model_bin)

    def score_path(self, path: list[int], rid: int | None = None) -> list[float]:
        if self.model is None:
            return [self.default_score] * len(path)

        scores = []
        for t in range(len(path)):
            text = _path_to_fasttext_line(path[: t + 1], position_tokens=self.position_tokens)
            labels, probs = self.model.predict(text, k=2)
            label_prob = dict(zip(labels, probs))
            scores.append(float(label_prob.get("__label__pass", 0.0)))
        return scores


# ── Count-based positional n-gram models ─────────────────────────────


def _positional_ngrams(path: list[int], word_ngrams: int = 5) -> list[str]:
    """Enumerate all positional n-gram keys for a cluster-ID prefix.

    Produces the same feature names as inspect_fasttext_ngrams (e.g.
    ``"c25 c0 c3 c4@0,1,2,3"``), including the ``</s>`` sentinel.
    """
    tokens = [f"c{c}" for c in path] + ["</s>"]
    out: list[str] = []
    for n in range(1, word_ngrams + 1):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i : i + n])
            pos = ",".join(str(p) for p in range(i, i + n))
            out.append(f"{gram}@{pos}")
    return out


def _ngrams(path: list[int], word_ngrams: int = 5) -> list[str]:
    """Enumerate *non-positional* n-gram keys (position-independent).

    E.g. ``"c25 c0 c3 c4"`` — the same cluster subsequence at any
    position maps to the same feature.  Includes ``</s>`` sentinel.
    """
    tokens = [f"c{c}" for c in path] + ["</s>"]
    out: list[str] = []
    for n in range(1, word_ngrams + 1):
        for i in range(len(tokens) - n + 1):
            out.append(" ".join(tokens[i : i + n]))
    return out


class LRPositionalNgramPRM:
    """Logistic regression on n-gram features over cluster-ID paths.

    By default uses *non-positional* n-grams so that the same cluster
    subsequence at any step position maps to a single feature, keeping the
    vocabulary small.  Set ``positional=True`` to revert to position-tagged
    features (much larger vocabulary, needs more data).

    L2 regularization naturally shrinks low-support features toward zero.
    """

    def __init__(
        self,
        word_ngrams: int = 5,
        C: float = 1.0,
        prefix_training: bool = True,
        positional: bool = False,
    ):
        self.word_ngrams = word_ngrams
        self.C = C
        self.prefix_training = prefix_training
        self.positional = positional

    def fit(
        self,
        rollout_paths: dict[int, list[int]],
        rewards: dict[int, float],
        K: int,
        sample_weights: dict[int, float] | None = None,
    ):
        from sklearn.linear_model import LogisticRegression
        from scipy.sparse import lil_matrix

        self.K = K
        labels_list = [
            1.0 if rewards.get(r, 0) >= 0.5 else 0.0 for r in rollout_paths
        ]
        if len(set(labels_list)) < 2:
            self.model = None
            self.vocab: dict[str, int] = {}
            self.default_score = float(labels_list[0]) if labels_list else 0.5
            return

        rows: list[dict[str, int]] = []
        y_list: list[float] = []
        sw_list: list[float] = []

        for rid, path in rollout_paths.items():
            if len(path) == 0:
                continue
            label = 1.0 if rewards.get(rid, 0) >= 0.5 else 0.0
            base_w = sample_weights.get(rid, 1.0) if sample_weights else 1.0
            prefixes = range(1, len(path) + 1) if self.prefix_training else [len(path)]
            per_w = base_w / len(path) if self.prefix_training else base_w
            ngram_fn = _positional_ngrams if self.positional else _ngrams
            for t in prefixes:
                feats: dict[str, int] = {}
                for key in ngram_fn(path[:t], self.word_ngrams):
                    feats[key] = feats.get(key, 0) + 1
                rows.append(feats)
                y_list.append(label)
                sw_list.append(per_w)

        vocab: dict[str, int] = {}
        for r in rows:
            for k in r:
                if k not in vocab:
                    vocab[k] = len(vocab)
        self.vocab = vocab

        X = lil_matrix((len(rows), len(vocab)), dtype=np.float64)
        for i, r in enumerate(rows):
            for k, v in r.items():
                X[i, vocab[k]] = v
        X = X.tocsr()

        y = np.array(y_list)
        sw = np.array(sw_list)

        self.model = LogisticRegression(
            C=self.C, l1_ratio=0, class_weight="balanced",
            solver="lbfgs", max_iter=1000,
        )
        self.model.fit(X, y, sample_weight=sw)
        self.default_score = 0.5

    def _featurize(self, path: list[int]):
        from scipy.sparse import lil_matrix
        ngram_fn = _positional_ngrams if self.positional else _ngrams
        X = lil_matrix((1, len(self.vocab)), dtype=np.float64)
        for key in ngram_fn(path, self.word_ngrams):
            idx = self.vocab.get(key)
            if idx is not None:
                X[0, idx] += 1
        return X.tocsr()

    def score_path(self, path: list[int], rid: int | None = None) -> list[float]:
        if self.model is None or not self.vocab:
            return [self.default_score] * max(len(path), 1)
        scores = []
        for t in range(len(path)):
            X = self._featurize(path[: t + 1])
            prob = self.model.predict_proba(X)[0]
            idx_1 = list(self.model.classes_).index(1)
            scores.append(float(prob[idx_1]))
        return scores


class MarkovNgramPRM:
    """Higher-order Markov PRM with Katz-style backoff smoothing.

    Extends ``MarkovAbsorptionPRM`` from unigram ``(cluster, step)`` to
    n-gram ``(cluster_sequence, step)`` contexts.  At scoring time the
    highest-order context with sufficient support is used, backing off to
    shorter contexts (and ultimately the global rate) when data is sparse.
    """

    def __init__(
        self,
        max_order: int = 3,
        smoothing: float = 1.0,
        min_count: int = 3,
    ):
        self.max_order = max_order
        self.smoothing = smoothing
        self.min_count = min_count

    def fit(
        self,
        rollout_paths: dict[int, list[int]],
        rewards: dict[int, float],
        K: int,
        sample_weights: dict[int, float] | None = None,
    ):
        self.K = K
        # Per-cluster global counts (for unigram backoff prior)
        cluster_pass = np.zeros(K)
        cluster_count = np.zeros(K)

        # ngram_pass[(context_tuple, end_step)] = weighted pass count
        ngram_pass: dict[tuple[tuple[int, ...], int], float] = {}
        ngram_count: dict[tuple[tuple[int, ...], int], float] = {}

        for rid, path in rollout_paths.items():
            w = sample_weights.get(rid, 1.0) if sample_weights else 1.0
            passed = rewards.get(rid, 0) >= 0.5
            for t, c in enumerate(path):
                cluster_count[c] += w
                if passed:
                    cluster_pass[c] += w
                for order in range(1, self.max_order + 1):
                    start = max(0, t - order + 1)
                    if t - start + 1 < order:
                        continue
                    ctx = tuple(path[start : t + 1])
                    key = (ctx, t)
                    ngram_count[key] = ngram_count.get(key, 0) + w
                    if passed:
                        ngram_pass[key] = ngram_pass.get(key, 0) + w

        s = self.smoothing
        self.cluster_rate = (cluster_pass + s * 0.5) / (cluster_count + s)
        self.ngram_pass = ngram_pass
        self.ngram_count = ngram_count

    def score_path(self, path: list[int], rid: int | None = None) -> list[float]:
        s = self.smoothing
        scores = []
        for t in range(len(path)):
            score = self._score_step(path, t, s)
            scores.append(float(score))
        return scores

    def _unigram_score(self, path: list[int], t: int, s: float) -> float:
        """Unigram (order-1) score: P(pass | cluster, step) smoothed toward
        the global cluster rate — identical to MarkovAbsorptionPRM."""
        c = path[t]
        key = ((c,), t)
        n = self.ngram_count.get(key, 0)
        p = self.ngram_pass.get(key, 0)
        prior = float(self.cluster_rate[c]) if c < len(self.cluster_rate) else 0.5
        return (p + s * prior) / (n + s)

    def _score_step(self, path: list[int], t: int, s: float) -> float:
        """Compute P(pass) at step *t* with backoff from max_order to 1.

        Higher-order contexts (order >= 2) require at least ``min_count``
        observations; the order-1 unigram is always used (no gate), matching
        ``MarkovAbsorptionPRM`` behaviour at the base level.
        """
        for order in range(self.max_order, 1, -1):
            start = max(0, t - order + 1)
            if t - start + 1 < order:
                continue
            ctx = tuple(path[start : t + 1])
            key = (ctx, t)
            n = self.ngram_count.get(key, 0)
            if n < self.min_count:
                continue
            p = self.ngram_pass.get(key, 0)
            prior = self._backoff_prior(path, t, order - 1, s)
            return (p + s * prior) / (n + s)
        return self._unigram_score(path, t, s)

    def _backoff_prior(self, path: list[int], t: int, order: int, s: float) -> float:
        """Recursive prior from (order-1)-gram, bottoming out at unigram."""
        if order <= 1:
            return self._unigram_score(path, t, s)
        start = max(0, t - order + 1)
        if t - start + 1 < order:
            return self._unigram_score(path, t, s)
        ctx = tuple(path[start : t + 1])
        key = (ctx, t)
        n = self.ngram_count.get(key, 0)
        if n < self.min_count:
            return self._backoff_prior(path, t, order - 1, s)
        p = self.ngram_pass.get(key, 0)
        prior = self._backoff_prior(path, t, order - 1, s)
        return (p + s * prior) / (n + s)


class NaiveBayesPRM:
    """Bernoulli Naive Bayes on n-gram presence with Laplace smoothing.

    For each class (pass / fail), estimates P(n-gram present | class) from
    training prefixes.  Smoothing ``alpha`` acts as a pseudo-count that
    naturally regularises features with low support.

    By default uses non-positional n-grams (same subsequence at any step
    maps to one feature).  Set ``positional=True`` for position-tagged
    features.
    """

    def __init__(
        self,
        word_ngrams: int = 5,
        alpha: float = 1.0,
        prefix_training: bool = True,
        positional: bool = False,
    ):
        self.word_ngrams = word_ngrams
        self.alpha = alpha
        self.prefix_training = prefix_training
        self.positional = positional

    def fit(
        self,
        rollout_paths: dict[int, list[int]],
        rewards: dict[int, float],
        K: int,
        sample_weights: dict[int, float] | None = None,
    ):
        self.K = K
        labels_list = [
            1.0 if rewards.get(r, 0) >= 0.5 else 0.0 for r in rollout_paths
        ]
        if len(set(labels_list)) < 2:
            self.vocab: dict[str, int] = {}
            self.log_prior_pass = 0.0
            self.log_prior_fail = 0.0
            self.log_p_pass: np.ndarray = np.array([])
            self.log_p_fail: np.ndarray = np.array([])
            self.log_q_pass: np.ndarray = np.array([])
            self.log_q_fail: np.ndarray = np.array([])
            self.default_score = float(labels_list[0]) if labels_list else 0.5
            return

        # Collect per-class feature presence counts
        vocab: dict[str, int] = {}
        pass_counts: dict[str, float] = {}
        fail_counts: dict[str, float] = {}
        n_pass = 0.0
        n_fail = 0.0

        for rid, path in rollout_paths.items():
            if len(path) == 0:
                continue
            passed = rewards.get(rid, 0) >= 0.5
            base_w = sample_weights.get(rid, 1.0) if sample_weights else 1.0
            prefixes = range(1, len(path) + 1) if self.prefix_training else [len(path)]
            per_w = base_w / len(path) if self.prefix_training else base_w

            ngram_fn = _positional_ngrams if self.positional else _ngrams
            for t in prefixes:
                if passed:
                    n_pass += per_w
                else:
                    n_fail += per_w
                seen: set[str] = set()
                for key in ngram_fn(path[:t], self.word_ngrams):
                    if key not in seen:
                        seen.add(key)
                        if key not in vocab:
                            vocab[key] = len(vocab)
                        target = pass_counts if passed else fail_counts
                        target[key] = target.get(key, 0) + per_w

        self.vocab = vocab
        V = len(vocab)
        a = self.alpha

        # Class-balanced priors
        total = n_pass + n_fail
        self.log_prior_pass = float(np.log(max(n_pass / total, 1e-10)))
        self.log_prior_fail = float(np.log(max(n_fail / total, 1e-10)))

        # P(feature present | class) and P(feature absent | class)
        log_p_pass = np.zeros(V)
        log_p_fail = np.zeros(V)
        log_q_pass = np.zeros(V)
        log_q_fail = np.zeros(V)
        for feat, idx in vocab.items():
            cp = pass_counts.get(feat, 0)
            cf = fail_counts.get(feat, 0)
            pp = (cp + a) / (n_pass + 2 * a)
            pf = (cf + a) / (n_fail + 2 * a)
            log_p_pass[idx] = np.log(max(pp, 1e-15))
            log_p_fail[idx] = np.log(max(pf, 1e-15))
            log_q_pass[idx] = np.log(max(1 - pp, 1e-15))
            log_q_fail[idx] = np.log(max(1 - pf, 1e-15))

        # Precompute per-feature log-odds delta for switching from absent→present
        self.delta = (log_p_pass - log_q_pass) - (log_p_fail - log_q_fail)
        # All-absent baseline log-odds
        self.base_log_odds = (
            self.log_prior_pass + float(log_q_pass.sum())
            - self.log_prior_fail - float(log_q_fail.sum())
        )
        self.default_score = 0.5

    def score_path(self, path: list[int], rid: int | None = None) -> list[float]:
        if not self.vocab:
            return [self.default_score] * max(len(path), 1)
        ngram_fn = _positional_ngrams if self.positional else _ngrams
        scores = []
        for t in range(len(path)):
            present = set()
            for key in ngram_fn(path[: t + 1], self.word_ngrams):
                idx = self.vocab.get(key)
                if idx is not None:
                    present.add(idx)
            log_odds = self.base_log_odds + sum(
                self.delta[idx] for idx in present
            )
            scores.append(_sigmoid(log_odds))
        return scores


class PerStepFastTextPRM:
    """Train a separate FastText model for every prefix length t.

    For each t in 1..T_max, model f_t is trained on
        { (s_{1:t}^{(i)}, y_i) : T_i >= t }
    i.e. exactly the prefixes of length t from rollouts that survived to t.
    At score time, f_t is queried on s_{1:t}.

    Step buckets (`step_bucket > 1`) merge consecutive step lengths into the
    same model — e.g. step_bucket=3 trains one model for t in {1,2,3}, one for
    {4,5,6}, etc.  This shares statistical strength across nearby steps and
    cuts training cost ~step_bucket-fold.
    """

    def __init__(
        self,
        wordNgrams: int = 5,
        dim: int = 20,
        epoch: int = 25,
        lr: float = 0.5,
        position_tokens: bool = False,
        step_bucket: int = 1,
    ):
        self.ft_kwargs = dict(
            wordNgrams=wordNgrams, dim=dim, epoch=epoch, lr=lr,
            position_tokens=position_tokens,
        )
        self.step_bucket = max(1, int(step_bucket))

    def _bucket(self, t: int) -> int:
        return ((t - 1) // self.step_bucket) * self.step_bucket + 1

    def fit(
        self,
        rollout_paths: dict[int, list[int]],
        rewards: dict[int, float],
        K: int,
        sample_weights: dict[int, float] | None = None,
    ):
        from collections import defaultdict

        self.K = K
        T_max = max((len(p) for p in rollout_paths.values()), default=0)
        if T_max == 0:
            self.models = {}
            self.default_score = 0.5
            return

        bucket_paths: dict[int, dict[int, list[int]]] = defaultdict(dict)
        bucket_rewards: dict[int, dict[int, float]] = defaultdict(dict)
        bucket_weights: dict[int, dict[int, float]] = defaultdict(dict)
        for rid, path in rollout_paths.items():
            T = len(path)
            if T == 0:
                continue
            for t in range(1, T + 1):
                bkey = self._bucket(t)
                if rid in bucket_paths[bkey]:
                    continue
                bucket_paths[bkey][rid] = path[:t]
                bucket_rewards[bkey][rid] = rewards.get(rid, 0)
                if sample_weights is not None:
                    bucket_weights[bkey][rid] = sample_weights.get(rid, 1.0)

        self.models: dict[int, FastTextPRM] = {}
        self.fallback_score = float(np.mean([
            1.0 if rewards.get(r, 0) >= 0.5 else 0.0
            for r in rollout_paths
        ])) if rollout_paths else 0.5

        for bkey, paths_b in bucket_paths.items():
            if len(paths_b) < 2:
                continue
            labels_b = set(
                "pass" if bucket_rewards[bkey][r] >= 0.5 else "fail"
                for r in paths_b
            )
            prm_b = FastTextPRM(**self.ft_kwargs)
            sw = bucket_weights[bkey] if sample_weights is not None else None
            prm_b.fit(paths_b, bucket_rewards[bkey], K, sample_weights=sw)
            self.models[bkey] = prm_b

        self.T_max = T_max

    def score_path(self, path: list[int], rid: int | None = None) -> list[float]:
        if not self.models:
            return [self.fallback_score] * len(path)

        avail = sorted(self.models.keys())
        scores: list[float] = []
        for t in range(1, len(path) + 1):
            bkey = self._bucket(t)
            prm_b = self.models.get(bkey)
            if prm_b is None:
                bkey = min(avail, key=lambda x: abs(x - bkey))
                prm_b = self.models[bkey]
            if prm_b.model is None:
                scores.append(prm_b.default_score)
                continue
            text = _path_to_fasttext_line(
                path[:t], position_tokens=prm_b.position_tokens
            )
            ft_labels, probs = prm_b.model.predict(text, k=2)
            label_prob = dict(zip(ft_labels, probs))
            scores.append(float(label_prob.get("__label__pass", 0.0)))
        return scores


# ── Level 4: GRU sequence model ─────────────────────────────────────


class GRUPRM:
    """Small GRU over cluster paths with position as a continuous feature."""

    def __init__(
        self,
        embed_dim: int = 8,
        hidden_dim: int = 16,
        epochs: int = 200,
        lr: float = 0.01,
        weight_decay: float = 1e-3,
    ):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

    def fit(
        self,
        rollout_paths: dict[int, list[int]],
        rewards: dict[int, float],
        K: int,
        sample_weights: dict[int, float] | None = None,
    ):
        import torch
        import torch.nn as nn

        self.K = K
        labels = [
            "pass" if rewards.get(r, 0) >= 0.5 else "fail"
            for r in rollout_paths
        ]
        if len(set(labels)) < 2:
            self.model = None
            self.default_score = 1.0 if labels[0] == "pass" else 0.0
            return

        max_len = max(len(p) for p in rollout_paths.values())
        rids = list(rollout_paths.keys())
        n = len(rids)

        cluster_seqs = torch.zeros(n, max_len, dtype=torch.long)
        pos_seqs = torch.zeros(n, max_len, dtype=torch.float32)
        targets = torch.zeros(n, dtype=torch.float32)
        lengths = torch.zeros(n, dtype=torch.long)

        for i, rid in enumerate(rids):
            path = rollout_paths[rid]
            L = len(path)
            lengths[i] = L
            cluster_seqs[i, :L] = torch.tensor(path, dtype=torch.long)
            pos_seqs[i, :L] = torch.tensor(
                [t / max(max_len, 1) for t in range(L)], dtype=torch.float32
            )
            targets[i] = 1.0 if rewards.get(rid, 0) >= 0.5 else 0.0

        embed = nn.Embedding(K, self.embed_dim)
        gru = nn.GRU(
            input_size=self.embed_dim + 1,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        head = nn.Linear(self.hidden_dim, 1)

        params = list(embed.parameters()) + list(gru.parameters()) + list(head.parameters())
        optimizer = torch.optim.Adam(
            params, lr=self.lr, weight_decay=self.weight_decay
        )

        embed.train()
        gru.train()
        head.train()

        for _ in range(self.epochs):
            emb = embed(cluster_seqs)
            inp = torch.cat([emb, pos_seqs.unsqueeze(-1)], dim=-1)
            out, _ = gru(inp)

            idx = (lengths - 1).clamp(min=0).unsqueeze(-1).unsqueeze(-1)
            idx = idx.expand(-1, -1, self.hidden_dim)
            last_hidden = out.gather(1, idx).squeeze(1)

            logits = head(last_hidden).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        embed.eval()
        gru.eval()
        head.eval()
        self._embed = embed
        self._gru = gru
        self._head = head
        self._max_len = max_len
        self.model = True

    def score_path(self, path: list[int], rid: int | None = None) -> list[float]:
        if self.model is None:
            return [self.default_score] * len(path)

        import torch

        scores = []
        with torch.no_grad():
            for t in range(len(path)):
                prefix = path[: t + 1]
                L = len(prefix)
                c = torch.tensor(prefix, dtype=torch.long).unsqueeze(0)
                p = torch.tensor(
                    [i / max(self._max_len, 1) for i in range(L)],
                    dtype=torch.float32,
                ).unsqueeze(0)
                emb = self._embed(c)
                inp = torch.cat([emb, p.unsqueeze(-1)], dim=-1)
                out, _ = self._gru(inp)
                logit = self._head(out[:, -1, :]).squeeze(-1)
                scores.append(float(torch.sigmoid(logit).item()))
        return scores


class TransformerPRM:
    """Bidirectional transformer with prefix training on interleaved action/response embeddings.

    Operates directly on continuous step embeddings (bypasses clustering).
    Each step produces two tokens: action, then response.
    Trained on all prefix lengths for cross-task generalization.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.2,
        max_steps: int = 100,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 128,
        warmup_frac: float = 0.05,
        prefix_training: bool = True,
        device: str | None = None,
        log_every: int = 50,
        no_pos: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.max_steps = max_steps
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.warmup_frac = warmup_frac
        self.prefix_training = prefix_training
        self.device = device
        self.log_every = log_every
        # no_pos: ablation — drop the position embedding so the encoder is
        # permutation-invariant (bag-of-steps). Retains the action/response
        # type flag; loses all step-ordering information.
        self.no_pos = no_pos
        self.model = None

    @staticmethod
    def _select_device():
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit_from_embeddings(
        self,
        act_sequences: list[np.ndarray],
        res_sequences: list[np.ndarray],
        labels: list[int],
        epoch_callback: callable | None = None,
    ):
        """Train on interleaved action/response embedding sequences.

        Args:
            act_sequences: list of (T_i, D) arrays, one per rollout
            res_sequences: list of (T_i, D) arrays, one per rollout
            labels: list of 0/1 per rollout
            epoch_callback: optional fn(prm, epoch, loss) called after each epoch
        """
        import torch
        import torch.nn as nn
        import math

        device = torch.device(self.device) if self.device else self._select_device()
        if not getattr(TransformerPRM, "_device_printed", False):
            print(f"  TransformerPRM training on device: {device}")
            TransformerPRM._device_printed = True

        if len(set(labels)) < 2:
            self.model = None
            self.default_score = float(labels[0]) if labels else 0.5
            return

        d_emb = act_sequences[0].shape[1]
        d_input = d_emb + 1  # +1 for scalar type flag (0=action, 1=response)

        prefixes = []
        prefix_positions = []
        prefix_labels = []
        for act_seq, res_seq, label in zip(act_sequences, res_sequences, labels):
            # Truncate outlier-length rollouts: positions clip at max_steps
            # anyway, and one 1600-step rollout in a padded batch OOMs MPS.
            act_seq = act_seq[: self.max_steps]
            res_seq = res_seq[: self.max_steps]
            T = act_seq.shape[0]
            interleaved = np.empty((2 * T, d_input), dtype=np.float32)
            positions = np.empty(2 * T, dtype=np.int64)
            for t in range(T):
                interleaved[2 * t, :d_emb] = act_seq[t]
                interleaved[2 * t, d_emb] = 0.0
                interleaved[2 * t + 1, :d_emb] = res_seq[t]
                interleaved[2 * t + 1, d_emb] = 1.0
                positions[2 * t] = t
                positions[2 * t + 1] = t
            if self.prefix_training:
                for length in range(1, 2 * T + 1):
                    prefixes.append(interleaved[:length])
                    prefix_positions.append(positions[:length])
                    prefix_labels.append(label)
            else:
                # Full-length sequences only (matches ProtoNet Table 2 episodes)
                prefixes.append(interleaved)
                prefix_positions.append(positions)
                prefix_labels.append(label)

        proj = nn.Linear(d_input, self.hidden_dim).to(device)
        pos_emb = nn.Embedding(self.max_steps, self.hidden_dim).to(device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            batch_first=True,
        )
        # enable_nested_tensor=False: the nested-tensor fast path segfaults on
        # MPS for large masked batches.
        encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers,
            enable_nested_tensor=False,
        ).to(device)
        head = nn.Linear(self.hidden_dim, 1).to(device)

        all_params = (
            list(proj.parameters())
            + list(pos_emb.parameters())
            + list(encoder.parameters())
            + list(head.parameters())
        )
        optimizer = torch.optim.AdamW(
            all_params, lr=self.lr, weight_decay=self.weight_decay,
        )

        n_examples = len(prefixes)
        n_batches = max(1, n_examples // self.batch_size)
        total_steps = self.epochs * n_batches
        warmup_steps = int(total_steps * self.warmup_frac)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        order = np.arange(n_examples)
        for epoch in range(self.epochs):
            np.random.shuffle(order)
            proj.train(); pos_emb.train()
            encoder.train(); head.train()

            epoch_loss = 0.0
            epoch_batches = 0
            for batch_start in range(0, n_examples, self.batch_size):
                batch_idx = order[batch_start:batch_start + self.batch_size]
                max_len = max(prefixes[i].shape[0] for i in batch_idx)
                B = len(batch_idx)

                tokens_pad = torch.zeros(B, max_len, d_input, device=device)
                pos_pad = torch.zeros(B, max_len, dtype=torch.long, device=device)
                mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
                targets = torch.zeros(B, device=device)

                for j, idx in enumerate(batch_idx):
                    L = prefixes[idx].shape[0]
                    tokens_pad[j, :L] = torch.from_numpy(prefixes[idx]).to(device)
                    pos_pad[j, :L] = torch.from_numpy(
                        np.minimum(prefix_positions[idx], self.max_steps - 1)
                    ).to(device)
                    mask[j, :L] = False
                    targets[j] = prefix_labels[idx]

                h = proj(tokens_pad)
                if not self.no_pos:
                    h = h + pos_emb(pos_pad)
                enc_out = encoder(h, src_key_padding_mask=mask)

                valid = (~mask).unsqueeze(-1).float()
                pooled = (enc_out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
                logits = head(pooled).squeeze(-1)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                epoch_batches += 1

            avg_loss = epoch_loss / epoch_batches
            if (epoch + 1) % self.log_every == 0 or epoch + 1 == self.epochs:
                print(f"    epoch {epoch+1}/{self.epochs}  loss={avg_loss:.4f}",
                      flush=True)

            if epoch_callback is not None:
                proj.eval(); pos_emb.eval()
                encoder.eval(); head.eval()
                self._proj = proj
                self._pos_emb = pos_emb
                self._encoder = encoder
                self._head = head
                self._d_emb = d_emb
                self._d_input = d_input
                self._device = device
                self.model = True
                epoch_callback(self, epoch + 1, avg_loss)

        proj.eval(); pos_emb.eval()
        encoder.eval(); head.eval()
        self._device = device
        self._proj = proj
        self._pos_emb = pos_emb
        self._encoder = encoder
        self._head = head
        self._d_emb = d_emb
        self._d_input = d_input
        self.model = True

    def fit(
        self,
        rollout_paths: dict[int, list[int]],
        rewards: dict[int, float],
        K: int,
        sample_weights: dict[int, float] | None = None,
        *,
        act_embeddings: dict[int, np.ndarray] | None = None,
        res_embeddings: dict[int, np.ndarray] | None = None,
    ):
        """Standard PRM interface. Requires act/res embeddings passed via kwargs."""
        if act_embeddings is None or res_embeddings is None:
            self.model = None
            self.default_score = 0.5
            return

        act_seqs, res_seqs, labels = [], [], []
        for rid in rollout_paths:
            if rid not in act_embeddings or rid not in res_embeddings:
                continue
            act_seqs.append(act_embeddings[rid])
            res_seqs.append(res_embeddings[rid])
            labels.append(int(rewards.get(rid, 0) >= 0.5))

        self.fit_from_embeddings(act_seqs, res_seqs, labels)

    def score_final(self, act_emb: np.ndarray, res_emb: np.ndarray) -> float:
        """P(pass) from the full-length sequence — single forward pass."""
        if self.model is None:
            return getattr(self, "default_score", 0.5)

        import torch

        device = self._device
        act_emb = act_emb[: self.max_steps]
        res_emb = res_emb[: self.max_steps]
        T = act_emb.shape[0]
        d_emb = self._d_emb
        d_input = self._d_input
        interleaved = np.empty((2 * T, d_input), dtype=np.float32)
        positions = np.empty(2 * T, dtype=np.int64)
        for t in range(T):
            interleaved[2 * t, :d_emb] = act_emb[t]
            interleaved[2 * t, d_emb] = 0.0
            interleaved[2 * t + 1, :d_emb] = res_emb[t]
            interleaved[2 * t + 1, d_emb] = 1.0
            positions[2 * t] = t
            positions[2 * t + 1] = t

        with torch.no_grad():
            tok = torch.from_numpy(interleaved).unsqueeze(0).to(device)
            pos = torch.from_numpy(
                np.minimum(positions, self.max_steps - 1)
            ).unsqueeze(0).to(device)
            h = self._proj(tok)
            if not self.no_pos:
                h = h + self._pos_emb(pos)
            enc_out = self._encoder(h)
            pooled = enc_out.mean(dim=1)
            logit = self._head(pooled).squeeze(-1)
            return float(torch.sigmoid(logit).item())

    def score_path(self, path: list[int], rid: int | None = None,
                   *, act_emb: np.ndarray | None = None,
                   res_emb: np.ndarray | None = None) -> list[float]:
        """Return P(pass) after each step (response token).

        Requires act_emb and res_emb arrays of shape (T, D).
        Falls back to default_score if embeddings are unavailable.
        """
        if self.model is None or act_emb is None or res_emb is None:
            default = getattr(self, "default_score", 0.5)
            return [default] * len(path)

        import torch

        device = self._device
        act_emb = act_emb[: self.max_steps]
        res_emb = res_emb[: self.max_steps]
        T = act_emb.shape[0]
        d_emb = self._d_emb
        d_input = self._d_input
        interleaved = np.empty((2 * T, d_input), dtype=np.float32)
        positions = np.empty(2 * T, dtype=np.int64)
        for t in range(T):
            interleaved[2 * t, :d_emb] = act_emb[t]
            interleaved[2 * t, d_emb] = 0.0
            interleaved[2 * t + 1, :d_emb] = res_emb[t]
            interleaved[2 * t + 1, d_emb] = 1.0
            positions[2 * t] = t
            positions[2 * t + 1] = t

        scores = []
        with torch.no_grad():
            for step in range(T):
                length = 2 * (step + 1)
                tok = torch.from_numpy(interleaved[:length]).unsqueeze(0).to(device)
                pos = torch.from_numpy(
                    np.minimum(positions[:length], self.max_steps - 1)
                ).unsqueeze(0).to(device)

                h = self._proj(tok)
                if not self.no_pos:
                    h = h + self._pos_emb(pos)
                enc_out = self._encoder(h)
                pooled = enc_out.mean(dim=1)
                logit = self._head(pooled).squeeze(-1)
                scores.append(float(torch.sigmoid(logit).item()))

        return scores


class _GradientReversal(object):
    """Gradient reversal layer for domain-adversarial training.

    Forward: identity.  Backward: negate gradient, scaled by lambda.
    Implemented as a hook rather than autograd.Function for simplicity.
    """

    @staticmethod
    def apply(x, lam: float):
        import torch

        out = x.clone()
        out.register_hook(lambda grad: -lam * grad)
        return out


class ProtoNetPRM:
    """Prototypical Network PRM with episodic meta-learning.

    Learns an embedding space where rollout success is predicted by
    Euclidean distance to class prototypes.  At inference time, a small
    support set (e.g. the first GRPO batch) defines the prototypes for
    a new task.

    Uses the same encoder architecture as TransformerPRM (projection +
    positional embedding + transformer encoder + masked mean pool) but
    replaces the BCE classification head with prototype-based distance
    classification trained episodically.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.3,
        max_steps: int = 100,
        epochs: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_frac: float = 0.05,
        k_support: int = 8,
        q_query: int = 8,
        episodes_per_epoch: int = 1000,
        lambda_adv: float = 0.0,
        lambda_adv_warmup_epochs: int = 10,
        no_pos: bool = False,
    ):
        self.no_pos = no_pos
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.max_steps = max_steps
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_frac = warmup_frac
        self.k_support = k_support
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch
        self.lambda_adv = lambda_adv
        self.lambda_adv_warmup_epochs = lambda_adv_warmup_epochs
        self.pooling = "mean"
        self.temperature = 1.0
        self.platt_a = 1.0
        self.platt_b = 0.0
        self.prefix_training = False
        self.model = None
        self._attn_query = None
        self._proto_pass = None
        self._proto_fail = None
        self._discriminator = None
        self._disc_n_domains = None

    @staticmethod
    def _select_device():
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _interleave_rollout(self, act_seq: np.ndarray, res_seq: np.ndarray):
        """Convert (T, D) action + response arrays into interleaved tokens."""
        d_emb = act_seq.shape[1]
        d_input = d_emb + 1
        T = act_seq.shape[0]
        interleaved = np.empty((2 * T, d_input), dtype=np.float32)
        positions = np.empty(2 * T, dtype=np.int64)
        for t in range(T):
            interleaved[2 * t, :d_emb] = act_seq[t]
            interleaved[2 * t, d_emb] = 0.0
            interleaved[2 * t + 1, :d_emb] = res_seq[t]
            interleaved[2 * t + 1, d_emb] = 1.0
            positions[2 * t] = t
            positions[2 * t + 1] = t
        return interleaved, positions

    def _encode_batch(self, tokens_list, positions_list):
        """Encode a batch of variable-length token sequences -> (B, hidden_dim).

        tokens_list: list of (L_i, d_input) arrays
        positions_list: list of (L_i,) arrays
        Returns: torch tensor (B, hidden_dim) on self._device
        """
        import torch

        device = self._device
        B = len(tokens_list)
        max_len = max(t.shape[0] for t in tokens_list)
        d_input = tokens_list[0].shape[1]

        tokens_pad = torch.zeros(B, max_len, d_input, device=device)
        pos_pad = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask = torch.ones(B, max_len, dtype=torch.bool, device=device)

        for j, (tok, pos) in enumerate(zip(tokens_list, positions_list)):
            L = tok.shape[0]
            tokens_pad[j, :L] = torch.from_numpy(tok).to(device)
            pos_pad[j, :L] = torch.from_numpy(
                np.minimum(pos, self.max_steps - 1)
            ).to(device)
            mask[j, :L] = False

        h = self._proj(tokens_pad)
        if not getattr(self, "no_pos", False):
            h = h + self._pos_emb(pos_pad)
        enc_out = self._encoder(h, src_key_padding_mask=mask)

        if self.pooling == "attention" and self._attn_query is not None:
            # Attention pooling: learned query attends over encoder outputs
            scores = (enc_out * self._attn_query).sum(dim=-1)  # (B, L)
            scores = scores.masked_fill(mask, float('-inf'))
            weights = torch.softmax(scores, dim=-1)  # (B, L)
            pooled = (weights.unsqueeze(-1) * enc_out).sum(dim=1)  # (B, H)
            self._last_attn_weights = weights.detach()
        else:
            valid = (~mask).unsqueeze(-1).float()
            pooled = (enc_out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            self._last_attn_weights = None
        return pooled

    def fit_from_embeddings(
        self,
        act_sequences: list[np.ndarray],
        res_sequences: list[np.ndarray],
        labels: list[int],
        task_ids: list[str],
        epoch_callback: callable | None = None,
        model_ids: list[int] | None = None,
    ):
        """Episodic meta-training on interleaved action/response sequences.

        Args:
            act_sequences: list of (T_i, D) arrays, one per rollout
            res_sequences: list of (T_i, D) arrays, one per rollout
            labels: list of 0/1 per rollout
            task_ids: list of task identifier strings, one per rollout
            epoch_callback: optional fn(prm, epoch, loss) called after each epoch
            model_ids: optional list of 0/1 per rollout (e.g. 0=Qwen, 1=DeepSeek).
                       When provided with lambda_adv > 0, enables domain-adversarial
                       training with a gradient reversal layer.
        """
        import torch
        import torch.nn as nn
        import math

        device = self._select_device()
        print(f"  ProtoNetPRM training on device: {device}")

        if len(set(labels)) < 2:
            self.model = None
            self.default_score = float(labels[0]) if labels else 0.5
            return

        use_adversarial = (
            model_ids is not None
            and self.lambda_adv > 0
            and len(set(model_ids)) >= 2
        )

        # Pre-compute interleaved tokens for all rollouts
        all_tokens = []
        all_positions = []
        for act_seq, res_seq in zip(act_sequences, res_sequences):
            tok, pos = self._interleave_rollout(act_seq, res_seq)
            all_tokens.append(tok)
            all_positions.append(pos)

        # Group rollouts by task
        task_to_indices: dict[str, dict[int, list[int]]] = {}
        for i, (tid, lab) in enumerate(zip(task_ids, labels)):
            if tid not in task_to_indices:
                task_to_indices[tid] = {0: [], 1: []}
            task_to_indices[tid][lab].append(i)

        # Filter eligible tasks: need both classes
        eligible_tasks = []
        for tid, class_indices in task_to_indices.items():
            if len(class_indices[0]) >= 1 and len(class_indices[1]) >= 1:
                eligible_tasks.append(tid)

        if not eligible_tasks:
            print("  WARNING: no eligible tasks for episodic training")
            self.model = None
            self.default_score = 0.5
            return

        print(f"  {len(eligible_tasks)} eligible tasks for episodic training "
              f"(of {len(task_to_indices)} total)")

        d_emb = act_sequences[0].shape[1]
        d_input = d_emb + 1

        # Build encoder (same arch as TransformerPRM, no classification head)
        proj = nn.Linear(d_input, self.hidden_dim).to(device)
        pos_emb = nn.Embedding(self.max_steps, self.hidden_dim).to(device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            batch_first=True,
        )
        encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers,
            enable_nested_tensor=False,  # nested-tensor fast path unimplemented on MPS
        ).to(device)

        self._proj = proj
        self._pos_emb = pos_emb
        self._encoder = encoder
        self._d_emb = d_emb
        self._d_input = d_input
        self._device = device

        if self.pooling == "attention":
            attn_query = nn.Parameter(torch.randn(self.hidden_dim, device=device) * 0.02)
            self._attn_query = attn_query
        else:
            self._attn_query = None

        all_params = (
            list(proj.parameters())
            + list(pos_emb.parameters())
            + list(encoder.parameters())
        )
        if self._attn_query is not None:
            all_params.append(self._attn_query)

        # Domain-adversarial discriminator (multi-class over policy models)
        discriminator = None
        disc_batch_size = 16
        if use_adversarial:
            uniq = sorted(set(int(m) for m in model_ids))
            id2idx = {m: k for k, m in enumerate(uniq)}
            n_domains = len(uniq)
            self._disc_n_domains = n_domains
            discriminator = nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_domains),
            ).to(device)
            self._discriminator = discriminator
            all_params += list(discriminator.parameters())
            # Remap possibly-noncontiguous model ids to class indices 0..K-1.
            all_model_ids = np.array([id2idx[int(m)] for m in model_ids],
                                     dtype=np.int64)
            all_indices = np.arange(len(model_ids))
            counts = {uniq[k]: int((all_model_ids == k).sum())
                      for k in range(n_domains)}
            print(f"  Adversarial training: lambda_adv={self.lambda_adv}, "
                  f"warmup={self.lambda_adv_warmup_epochs} epochs, "
                  f"{n_domains} domains, counts by model_id: {counts}")

        optimizer = torch.optim.AdamW(
            all_params, lr=self.lr, weight_decay=self.weight_decay,
        )

        total_steps = self.epochs * self.episodes_per_epoch
        warmup_steps = int(total_steps * self.warmup_frac)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        rng = np.random.RandomState(42)

        for epoch in range(self.epochs):
            proj.train(); pos_emb.train(); encoder.train()
            if discriminator is not None:
                discriminator.train()
            epoch_loss = 0.0
            epoch_disc_loss = 0.0

            # Lambda schedule: linear ramp from 0 to lambda_adv
            if use_adversarial and self.lambda_adv_warmup_epochs > 0:
                lam = self.lambda_adv * min(
                    1.0, epoch / self.lambda_adv_warmup_epochs
                )
            else:
                lam = self.lambda_adv if use_adversarial else 0.0

            for _ep in range(self.episodes_per_epoch):
                tid = eligible_tasks[rng.randint(len(eligible_tasks))]
                cls_idx = task_to_indices[tid]

                # Sample K support + Q query per class (with replacement if needed)
                support_idx, query_idx = [], []
                support_labels, query_labels = [], []
                for c in [0, 1]:
                    pool = cls_idx[c]
                    n_need = self.k_support + self.q_query
                    replace = len(pool) < n_need
                    chosen = rng.choice(pool, size=n_need, replace=replace)
                    support_idx.extend(chosen[:self.k_support])
                    support_labels.extend([c] * self.k_support)
                    query_idx.extend(chosen[self.k_support:])
                    query_labels.extend([c] * self.q_query)

                # Encode support (optionally prefix-truncated) and query
                if hasattr(self, 'support_fraction') and self.support_fraction is not None:
                    s_tokens, s_positions = [], []
                    for i in support_idx:
                        full_len = all_tokens[i].shape[0]
                        T = full_len // 2
                        l = max(1, int(T * self.support_fraction))
                        s_tokens.append(all_tokens[i][:2 * l])
                        s_positions.append(all_positions[i][:2 * l])
                else:
                    s_tokens = [all_tokens[i] for i in support_idx]
                    s_positions = [all_positions[i] for i in support_idx]

                if self.prefix_training:
                    q_tokens, q_positions = [], []
                    for i in query_idx:
                        full_len = all_tokens[i].shape[0]
                        T = full_len // 2
                        if hasattr(self, 'prefix_fraction') and self.prefix_fraction is not None:
                            l = max(1, int(T * self.prefix_fraction))
                        else:
                            l = rng.randint(1, T + 1)
                        q_tokens.append(all_tokens[i][:2 * l])
                        q_positions.append(all_positions[i][:2 * l])
                else:
                    q_tokens = [all_tokens[i] for i in query_idx]
                    q_positions = [all_positions[i] for i in query_idx]

                s_embeds = self._encode_batch(s_tokens, s_positions)
                q_embeds = self._encode_batch(q_tokens, q_positions)

                # Compute prototypes
                s_labels_t = torch.tensor(support_labels, device=device)
                proto_pass = s_embeds[s_labels_t == 1].mean(dim=0)
                proto_fail = s_embeds[s_labels_t == 0].mean(dim=0)
                prototypes = torch.stack([proto_fail, proto_pass])  # (2, H)

                # Squared Euclidean distances: (Q_total, 2)
                # Manual computation to avoid torch.cdist (not on MPS)
                # eps: grad of sqrt blows up at 0; a deterministic encoder (e.g. no
                # dropout / no transformer) can put a query exactly on a prototype
                # when episodes sample with replacement -> NaN loss without it.
                diff = q_embeds.unsqueeze(1) - prototypes.unsqueeze(0)  # (Q, 2, H)
                dists = ((diff ** 2).sum(dim=-1) + 1e-8).sqrt()  # (Q, 2)
                logits = -dists  # negative distance -> higher = closer

                q_labels_t = torch.tensor(query_labels, dtype=torch.long,
                                          device=device)
                loss_proto = nn.functional.cross_entropy(logits, q_labels_t)

                # Domain-adversarial loss
                loss_disc = torch.tensor(0.0, device=device)
                if use_adversarial and lam > 0:
                    disc_idx = rng.choice(
                        all_indices, size=disc_batch_size, replace=False,
                    )
                    d_tokens = [all_tokens[i] for i in disc_idx]
                    d_positions = [all_positions[i] for i in disc_idx]
                    d_embeds = self._encode_batch(d_tokens, d_positions)

                    d_reversed = _GradientReversal.apply(d_embeds, lam)
                    d_logits = discriminator(d_reversed)          # (B, n_domains)
                    d_targets = torch.tensor(
                        all_model_ids[disc_idx], dtype=torch.long,
                        device=device,
                    )
                    loss_disc = nn.functional.cross_entropy(
                        d_logits, d_targets,
                    )

                loss = loss_proto + loss_disc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss_proto.item()
                epoch_disc_loss += loss_disc.item()

            avg_loss = epoch_loss / self.episodes_per_epoch
            if use_adversarial:
                avg_disc = epoch_disc_loss / self.episodes_per_epoch
                print(f"    epoch {epoch+1}/{self.epochs}  "
                      f"proto={avg_loss:.4f}  disc={avg_disc:.4f}  "
                      f"lam={lam:.4f}")
            else:
                print(f"    epoch {epoch+1}/{self.epochs}  loss={avg_loss:.4f}")

            if epoch_callback is not None:
                proj.eval(); pos_emb.eval(); encoder.eval()
                if discriminator is not None:
                    discriminator.eval()
                self.model = True
                epoch_callback(self, epoch + 1, avg_loss)
                proj.train(); pos_emb.train(); encoder.train()
                if discriminator is not None:
                    discriminator.train()

        proj.eval(); pos_emb.eval(); encoder.eval()
        if discriminator is not None:
            discriminator.eval()
        self.model = True

    def save(self, path: str | Path) -> None:
        """Persist encoder weights and config needed to reconstruct."""
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "proj": self._proj.state_dict(),
            "pos_emb": self._pos_emb.state_dict(),
            "encoder": self._encoder.state_dict(),
            "config": {
                "hidden_dim": self.hidden_dim,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "ffn_dim": self.ffn_dim,
                "dropout": self.dropout,
                "max_steps": self.max_steps,
                "d_emb": self._d_emb,
                "no_pos": getattr(self, "no_pos", False),
                "lambda_adv": self.lambda_adv,
                "lambda_adv_warmup_epochs": self.lambda_adv_warmup_epochs,
                "disc_n_domains": getattr(self, "_disc_n_domains", None),
                "pooling": self.pooling,
                "temperature": self.temperature,
                "platt_a": self.platt_a,
                "platt_b": self.platt_b,
            },
        }
        if self._attn_query is not None:
            payload["attn_query"] = self._attn_query.detach().cpu()
        if self._discriminator is not None:
            payload["discriminator"] = self._discriminator.state_dict()
        torch.save(payload, str(path))
        print(f"  ProtoNetPRM saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ProtoNetPRM":
        """Reconstruct a trained ProtoNetPRM from a checkpoint."""
        import torch
        import torch.nn as nn

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = ckpt["config"]

        prm = cls(
            hidden_dim=cfg["hidden_dim"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            ffn_dim=cfg.get("ffn_dim", 512),
            dropout=cfg.get("dropout", 0.0),
            max_steps=cfg.get("max_steps", 100),
            no_pos=cfg.get("no_pos", False),
            lambda_adv=cfg.get("lambda_adv", 0.0),
            lambda_adv_warmup_epochs=cfg.get("lambda_adv_warmup_epochs", 10),
        )
        device = cls._select_device()
        d_emb = cfg["d_emb"]
        d_input = d_emb + 1

        proj = nn.Linear(d_input, prm.hidden_dim).to(device)
        pos_emb = nn.Embedding(prm.max_steps, prm.hidden_dim).to(device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=prm.hidden_dim,
            nhead=prm.n_heads,
            dim_feedforward=prm.ffn_dim,
            dropout=prm.dropout,
            batch_first=True,
        )
        encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=prm.n_layers,
            enable_nested_tensor=False,  # nested-tensor fast path unimplemented on MPS
        ).to(device)

        proj.load_state_dict(ckpt["proj"])
        pos_emb.load_state_dict(ckpt["pos_emb"])
        encoder.load_state_dict(ckpt["encoder"])

        proj.eval(); pos_emb.eval(); encoder.eval()

        prm._proj = proj
        prm._pos_emb = pos_emb
        prm._encoder = encoder
        prm._d_emb = d_emb
        prm._d_input = d_input
        prm._device = device
        prm.pooling = cfg.get("pooling", "mean")
        prm.temperature = cfg.get("temperature", 1.0)
        prm.platt_a = cfg.get("platt_a", 1.0)
        prm.platt_b = cfg.get("platt_b", 0.0)

        if "attn_query" in ckpt:
            import torch.nn as nn
            prm._attn_query = nn.Parameter(ckpt["attn_query"].to(device))
            prm.pooling = "attention"

        prm.model = True

        if "discriminator" in ckpt:
            # Infer output size from the checkpoint (1 for old binary-BCE
            # discriminators, K for multi-class CE ones).
            sd = ckpt["discriminator"]
            n_out = cfg.get("disc_n_domains") or next(
                v.shape[0] for k, v in reversed(list(sd.items()))
                if k.endswith(".weight"))
            discriminator = nn.Sequential(
                nn.Linear(prm.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_out),
            ).to(device)
            discriminator.load_state_dict(sd)
            discriminator.eval()
            prm._discriminator = discriminator
            prm._disc_n_domains = n_out

        print(f"  ProtoNetPRM loaded from {path} (device={device})")
        return prm

    def _encode_rollout(
        self, act_emb: np.ndarray, res_emb: np.ndarray,
    ) -> np.ndarray:
        """Encode a single full rollout -> (hidden_dim,) numpy array."""
        import torch

        tok, pos = self._interleave_rollout(act_emb, res_emb)
        with torch.no_grad():
            embed = self._encode_batch([tok], [pos])
        return embed.squeeze(0).cpu().numpy()

    def set_support(
        self,
        act_sequences: list[np.ndarray],
        res_sequences: list[np.ndarray],
        labels: list[int],
    ):
        """Compute and store class prototypes from a support set.

        Called with the first GRPO batch at inference time.
        """
        if self.model is None:
            return

        embeddings = []
        for act_seq, res_seq in zip(act_sequences, res_sequences):
            embeddings.append(self._encode_rollout(act_seq, res_seq))

        embeddings = np.array(embeddings)
        labels_arr = np.array(labels)

        pass_mask = labels_arr == 1
        fail_mask = labels_arr == 0

        self._support_size = {0: int(fail_mask.sum()), 1: int(pass_mask.sum())}

        if pass_mask.any():
            self._proto_pass = embeddings[pass_mask].mean(axis=0)
        else:
            self._proto_pass = None

        if fail_mask.any():
            self._proto_fail = embeddings[fail_mask].mean(axis=0)
        else:
            self._proto_fail = None

    def _logits_from_embed(self, embed: np.ndarray) -> np.ndarray:
        """Return raw logits [-dist_fail, -dist_pass] from embedding."""
        if self._proto_pass is None:
            return np.array([-1e6, 0.0])
        if self._proto_fail is None:
            return np.array([0.0, -1e6])
        d_pass = float(np.linalg.norm(embed - self._proto_pass))
        d_fail = float(np.linalg.norm(embed - self._proto_fail))
        return np.array([-d_fail, -d_pass])

    def _score_from_embed(self, embed: np.ndarray) -> float:
        """Compute P(pass) from a single rollout embedding using prototypes."""
        logits = self._logits_from_embed(embed)
        if self.platt_a != 1.0 or self.platt_b != 0.0:
            logit_diff = logits[1] - logits[0]
            z = self.platt_a * logit_diff + self.platt_b
            z = float(np.clip(z, -30, 30))
            return 1.0 / (1.0 + np.exp(-z))
        logits = logits / self.temperature
        logits -= logits.max()
        exp_logits = np.exp(logits)
        return float(exp_logits[1] / exp_logits.sum())

    def score_final(
        self,
        *,
        act_emb: np.ndarray | None = None,
        res_emb: np.ndarray | None = None,
    ) -> float:
        """Return P(pass) for the full rollout (single forward pass)."""
        if self.model is None or act_emb is None or res_emb is None:
            return getattr(self, "default_score", 0.5)
        if self._proto_pass is None and self._proto_fail is None:
            return 0.5
        embed = self._encode_rollout(act_emb, res_emb)
        return self._score_from_embed(embed)

    def score_prefix(
        self,
        act_emb: np.ndarray,
        res_emb: np.ndarray,
    ) -> float:
        """Return P(pass) for a single rollout prefix. act_emb/res_emb are (T, D)."""
        if self.model is None or act_emb is None or res_emb is None:
            return getattr(self, "default_score", 0.5)
        if self._proto_pass is None and self._proto_fail is None:
            return 0.5
        embed = self._encode_rollout(act_emb, res_emb)
        return self._score_from_embed(embed)

    def score_path(
        self,
        path: list[int],
        rid: int | None = None,
        *,
        act_emb: np.ndarray | None = None,
        res_emb: np.ndarray | None = None,
    ) -> list[float]:
        """Return P(pass) after each step using prefix encoding.

        For step t, encodes act_emb[:t+1], res_emb[:t+1] and computes
        distance to prototypes.  Returns one score per entry in ``path``.
        """
        import torch

        if self.model is None or act_emb is None or res_emb is None:
            default = getattr(self, "default_score", 0.5)
            return [default] * len(path)

        if self._proto_pass is None and self._proto_fail is None:
            return [0.5] * len(path)

        T = act_emb.shape[0]
        scores = []
        with torch.no_grad():
            for step in range(T):
                prefix_act = act_emb[:step + 1]
                prefix_res = res_emb[:step + 1]
                embed = self._encode_rollout(prefix_act, prefix_res)
                scores.append(self._score_from_embed(embed))

        return scores


# ── Embedding baselines ──────────────────────────────────────────────


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(d / n) if n > 1e-10 else 0.0


# ── Causal embedding baselines (only use info up to step t) ──────────


class CausalEmbSimBaseline:
    """At step t, cosine similarity of step-t embedding to step-t success centroid."""

    def __init__(self, step_embeddings: np.ndarray, rid_to_rows: dict):
        self.emb = step_embeddings
        self.r2r = rid_to_rows

    def fit(self, rollout_paths: dict[int, list[int]], rewards: dict[int, float], K: int,
            sample_weights: dict[int, float] | None = None):
        success_rids = [rid for rid in rollout_paths if rewards.get(rid, 0) >= 0.5]
        success_rows = {rid: self.r2r.get(rid, []) for rid in success_rids}
        max_t = max(
            (len(rows) for rows in success_rows.values() if rows), default=0
        )
        d = self.emb.shape[1]
        self.step_centroids: list[np.ndarray | None] = []
        for t in range(max_t):
            embs = [
                self.emb[rows[t]]
                for rows in success_rows.values()
                if t < len(rows)
            ]
            if embs:
                self.step_centroids.append(np.mean(embs, axis=0))
            else:
                self.step_centroids.append(None)
        self._fallback = np.zeros(d)

    def score_path(self, path: list[int], rid: int | None = None) -> list[float]:
        if rid is None:
            return [0.5] * len(path)
        rows = self.r2r.get(rid, [])
        if not rows:
            return [0.5] * len(path)
        scores = []
        for t in range(len(path)):
            if t >= len(rows):
                scores.append(scores[-1] if scores else 0.5)
                continue
            centroid = (
                self.step_centroids[t]
                if t < len(self.step_centroids) and self.step_centroids[t] is not None
                else self._fallback
            )
            scores.append(_cosine_sim(self.emb[rows[t]], centroid))
        return scores


class CausalKnnBaseline:
    """At step t, k-NN on the step-t embedding against training rollouts at the same step."""

    def __init__(self, step_embeddings: np.ndarray, rid_to_rows: dict, k: int = 5):
        self.emb = step_embeddings
        self.r2r = rid_to_rows
        self.k = k

    def fit(self, rollout_paths: dict[int, list[int]], rewards: dict[int, float], K: int,
            sample_weights: dict[int, float] | None = None):
        self.train_rids = sorted(rollout_paths.keys())
        self.train_labels = {
            rid: 1.0 if rewards.get(rid, 0) >= 0.5 else 0.0
            for rid in self.train_rids
        }
        self.train_rid_rows = {rid: self.r2r.get(rid, []) for rid in self.train_rids}

    def score_path(self, path: list[int], rid: int | None = None) -> list[float]:
        if rid is None:
            return [0.5] * len(path)
        rows = self.r2r.get(rid, [])
        if not rows:
            return [0.5] * len(path)
        scores = []
        for t in range(len(path)):
            if t >= len(rows):
                scores.append(scores[-1] if scores else 0.5)
                continue
            test_emb = self.emb[rows[t]]
            test_norm = np.linalg.norm(test_emb)
            if test_norm < 1e-10:
                scores.append(0.5)
                continue
            train_embs = []
            train_labs = []
            for trid in self.train_rids:
                trows = self.train_rid_rows[trid]
                if t < len(trows):
                    train_embs.append(self.emb[trows[t]])
                    train_labs.append(self.train_labels[trid])
            if len(train_embs) < 2:
                scores.append(0.5)
                continue
            train_mat = np.stack(train_embs)
            train_labs_arr = np.array(train_labs)
            norms = np.linalg.norm(train_mat, axis=1, keepdims=True)
            train_normed = train_mat / np.maximum(norms, 1e-10)
            sims = train_normed @ (test_emb / test_norm)
            k = min(self.k, len(sims))
            top_k = np.argpartition(sims, -k)[-k:]
            scores.append(float(train_labs_arr[top_k].mean()))
        return scores


# ── Evaluation helpers ───────────────────────────────────────────────


def _aggregate_scores(step_scores: list[float], strategy: str) -> float:
    if strategy == "final":
        return step_scores[-1]
    elif strategy == "min":
        return min(step_scores)
    elif strategy == "mean":
        return float(np.mean(step_scores))
    raise ValueError(strategy)


def _evaluate_bon(
    test_rids: list[int],
    outcomes: dict[int, float],
    agg_scores: dict[int, float],
    n_values: list[int],
    rng: np.random.Generator,
    n_samples: int = 2000,
) -> dict[int, float]:
    rids = np.array(test_rids)
    out = np.array([outcomes[r] for r in rids])
    sc = np.array([agg_scores[r] for r in rids])
    M = len(rids)
    results = {}
    for N in n_values:
        if M == 0:
            results[N] = 0.0
            continue
        idx = rng.integers(0, M, size=(n_samples, N))
        best = idx[np.arange(n_samples), sc[idx].argmax(axis=1)]
        results[N] = float(out[best].mean())
    return results


def _evaluate_bon_filtered(
    test_rids: list[int],
    outcomes: dict[int, float],
    agg_scores: dict[int, float],
    n_values: list[int],
    rng: np.random.Generator,
    n_samples: int = 2000,
    oversample_factor: int = 5,
) -> dict[int, float]:
    """BoN evaluated only on samples containing at least one success and one failure.

    Returns the fraction of mixed samples where the PRM picks a success.
    Oracle is always 1.0, random baseline is (num_successes_in_sample / N).
    """
    rids = np.array(test_rids)
    out = np.array([outcomes[r] for r in rids])
    sc = np.array([agg_scores[r] for r in rids])
    M = len(rids)
    results = {}
    for N in n_values:
        if M == 0:
            results[N] = 0.0
            continue
        idx = rng.integers(0, M, size=(n_samples * oversample_factor, N))
        sample_outcomes = out[idx]
        has_pass = sample_outcomes.max(axis=1) >= 0.5
        has_fail = sample_outcomes.min(axis=1) < 0.5
        mixed = has_pass & has_fail
        idx_mixed = idx[mixed]
        if len(idx_mixed) == 0:
            results[N] = 0.0
            continue
        if len(idx_mixed) > n_samples:
            idx_mixed = idx_mixed[:n_samples]
        best = idx_mixed[np.arange(len(idx_mixed)), sc[idx_mixed].argmax(axis=1)]
        results[N] = float(out[best].mean())
    return results


def _evaluate_auroc(
    test_rids: list[int],
    outcomes: dict[int, float],
    agg_scores: dict[int, float],
) -> float:
    """AUROC on final aggregated scores (one per rollout)."""
    y = np.array([outcomes[r] for r in test_rids])
    s = np.array([agg_scores[r] for r in test_rids])
    if len(set(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, s))


# ── CV runner ────────────────────────────────────────────────────────

N_FOLDS = 5
N_REPEATS = 20
N_VALUES = [2, 4, 8, 16, 32]
STRATEGIES = ["final", "min", "mean"]


def run_cv_for_task(
    task: TaskData,
    prm_class,
    prm_kwargs: dict | None = None,
) -> dict:
    """5-fold CV × 20 repeats for one PRM level on one task."""
    if prm_kwargs is None:
        prm_kwargs = {}

    rids = np.array(task.all_rids)
    outcomes = {r: float(task.rewards.get(r, 0) >= 0.5) for r in task.all_rids}

    bon_acc = {s: {N: [] for N in N_VALUES} for s in STRATEGIES}
    auroc_acc = {s: [] for s in STRATEGIES}

    for rep in range(N_REPEATS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=rep * 1000 + 42)
        for fold, (train_idx, test_idx) in enumerate(kf.split(rids)):
            train_rids = rids[train_idx].tolist()
            test_rids_fold = rids[test_idx].tolist()

            train_paths = {r: task.rollout_paths[r] for r in train_rids}
            train_rewards = {r: task.rewards[r] for r in train_rids}

            prm = prm_class(**prm_kwargs)
            prm.fit(train_paths, train_rewards, task.K)

            step_scores: dict[int, list[float]] = {}
            for r in test_rids_fold:
                step_scores[r] = prm.score_path(task.rollout_paths[r], rid=r)

            rng = np.random.default_rng(rep * 100_000 + fold * 1000 + 7)
            for strat in STRATEGIES:
                agg = {r: _aggregate_scores(step_scores[r], strat) for r in test_rids_fold}
                bon = _evaluate_bon(test_rids_fold, outcomes, agg, N_VALUES, rng)
                for N in N_VALUES:
                    bon_acc[strat][N].append(bon[N])
                auroc_acc[strat].append(_evaluate_auroc(test_rids_fold, outcomes, agg))

    sr = float(np.mean(task.outcome))
    summary: dict = {
        "task": task.label,
        "n_rollouts": len(task.all_rids),
        "sr": sr,
        "K": task.K,
        "strategies": {},
    }

    for strat in STRATEGIES:
        strat_d: dict = {
            "auroc_mean": float(np.mean(auroc_acc[strat])),
            "auroc_std": float(np.std(auroc_acc[strat])),
            "bon": {},
        }
        for N in N_VALUES:
            vals = bon_acc[strat][N]
            strat_d["bon"][N] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "random": sr,
                "oracle": 1 - (1 - sr) ** N,
            }
        summary["strategies"][strat] = strat_d

    return summary


# ── Plotting ─────────────────────────────────────────────────────────

LEVEL_COLORS = {
    "level0_markov": "#a6cee3",
    "level1_likelihood": "#1f78b4",
    "level2_logreg": "#08519c",
    "level3_fasttext": "#33a02c",
    "level4_gru": "#ff7f00",
    "causal_emb_sim": "#fb9a99",
    "causal_knn": "#e31a1c",
}
LEVEL_LABELS = {
    "level0_markov": "L0 Markov",
    "level1_likelihood": "L1 Likelihood",
    "level2_logreg": "L2 LogReg",
    "level3_fasttext": "L3 fastText",
    "level4_gru": "L4 GRU",
    "causal_emb_sim": "Causal Cosine",
    "causal_knn": "Causal k-NN",
}
LEVEL_ORDER = ["level0_markov", "level1_likelihood", "level2_logreg", "level3_fasttext", "level4_gru"]
ALL_METHODS = ["level0_markov", "level1_likelihood", "level2_logreg", "level3_fasttext", "level4_gru",
               "causal_emb_sim", "causal_knn"]


def _short_label(task_label: str) -> str:
    """Shorten telecom labels for plotting."""
    if "telecom" not in task_label:
        return task_label.replace("/", " ")
    if "airplane" in task_label:
        return "telecom easy"
    if "bad_vpn" in task_label or "bad_network" in task_label:
        return "telecom hard"
    if "data_mode" in task_label:
        return "telecom data"
    return task_label[:20]


def _get_bon(res: dict, n: int) -> dict:
    """Get BoN entry handling int/str key ambiguity from JSON."""
    bon = res["strategies"]["final"]["bon"]
    return bon.get(n) or bon.get(str(n))


def plot_bon_per_task(
    task_label: str,
    level_results: dict[str, dict],
    out_path: Path,
):
    """BoN curves for one task across levels (strategy = final)."""
    sr = None
    fig, ax = plt.subplots(figsize=(7, 5))

    for level_name in ALL_METHODS:
        if level_name not in level_results:
            continue
        res = level_results[level_name]
        sr = res["sr"]
        bon_data = res["strategies"]["final"]["bon"]
        ns = sorted(int(k) for k in bon_data.keys())
        means = np.array([bon_data[n]["mean"] if n in bon_data else bon_data[str(n)]["mean"] for n in ns])
        stds = np.array([bon_data[n]["std"] if n in bon_data else bon_data[str(n)]["std"] for n in ns])

        ax.plot(ns, means, "o-", color=LEVEL_COLORS[level_name], label=LEVEL_LABELS[level_name])
        ax.fill_between(ns, means - stds, means + stds, alpha=0.12, color=LEVEL_COLORS[level_name])

    if sr is not None:
        ns_all = sorted(int(k) for k in list(level_results.values())[0]["strategies"]["final"]["bon"].keys())
        ax.axhline(sr, ls="--", color="gray", label=f"Random ({sr:.0%})")
        oracle = [1 - (1 - sr) ** N for N in ns_all]
        ax.plot(ns_all, oracle, "s--", color="black", alpha=0.5, label="Oracle")
        ax.set_xticks(ns_all)
        ax.set_xticklabels([str(n) for n in ns_all])

    ax.set_xlabel("N (Best-of-N)")
    ax.set_ylabel("Pass rate")
    ax.set_title(f"Best-of-N — {_short_label(task_label)}")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cross_task_summary(
    all_results: dict[str, dict[str, dict]],
    out_dir: Path,
):
    """Comprehensive summary figures across all tasks and levels."""
    tasks = sorted(all_results.keys())
    short = [_short_label(t) for t in tasks]
    n_tasks = len(tasks)
    first_task = all_results[tasks[0]]
    methods = [m for m in ALL_METHODS if m in first_task]
    levels = [l for l in LEVEL_ORDER if l in first_task]

    # ─── 1. Combined dashboard: BoN@8, AUROC, Normalized Lift ─────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    n_methods = len(methods)
    x = np.arange(n_tasks)
    width = 0.8 / max(n_methods, 1)

    # Panel A: BoN@8
    ax = axes[0]
    for i, m in enumerate(methods):
        vals = [_get_bon(all_results[t][m], 8)["mean"] for t in tasks]
        ax.bar(x + i * width, vals, width, label=LEVEL_LABELS.get(m, m),
               color=LEVEL_COLORS.get(m, "gray"))

    srs = [all_results[t][methods[0]]["sr"] for t in tasks]
    oracles = [1 - (1 - sr) ** 8 for sr in srs]
    mid = n_methods * width / 2
    for xi, o, sr in zip(x, oracles, srs):
        ax.plot(xi + mid, o, "kD", ms=5, zorder=5)
        ax.plot(xi + mid, sr, "_", color="gray", ms=10, mew=2, zorder=5)
    ax.plot([], [], "kD", ms=5, label="Oracle@8")
    ax.plot([], [], "_", color="gray", ms=8, mew=2, label="Random")
    ax.set_xticks(x + mid)
    ax.set_xticklabels(short, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Pass rate")
    ax.set_title("Best-of-8 Selection", fontweight="bold")
    ax.legend(fontsize=5.5, ncol=2, loc="lower left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.08)

    # Panel B: AUROC (final-step, per strategy = final)
    ax = axes[1]
    for i, m in enumerate(methods):
        vals = np.array([all_results[t][m]["strategies"]["final"]["auroc_mean"] for t in tasks])
        errs = np.array([all_results[t][m]["strategies"]["final"]["auroc_std"] for t in tasks])
        err_lo = np.minimum(errs, vals)
        err_hi = np.minimum(errs, 1.0 - vals)
        ax.bar(x + i * width, vals, width, yerr=[err_lo, err_hi], capsize=2,
               label=LEVEL_LABELS.get(m, m), color=LEVEL_COLORS.get(m, "gray"), alpha=0.85)
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5, lw=0.8)
    ax.set_xticks(x + mid)
    ax.set_xticklabels(short, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("Rollout-level AUROC (final score)", fontweight="bold")
    ax.legend(fontsize=5.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.35, 1.05)

    # Panel C: Normalized lift (PRM - random) / (oracle - random)
    ax = axes[2]
    for i, m in enumerate(methods):
        lifts = []
        for t in tasks:
            bon8 = _get_bon(all_results[t][m], 8)
            prm_val = bon8["mean"]
            rnd = bon8["random"]
            orc = bon8["oracle"]
            gap = orc - rnd
            lifts.append((prm_val - rnd) / gap if gap > 0.01 else 0.0)
        ax.bar(x + i * width, lifts, width, label=LEVEL_LABELS.get(m, m),
               color=LEVEL_COLORS.get(m, "gray"))
    ax.axhline(1, ls="--", color="black", alpha=0.3, lw=0.8, label="Oracle")
    ax.axhline(0, ls="-", color="gray", alpha=0.3, lw=0.8)
    ax.set_xticks(x + mid)
    ax.set_xticklabels(short, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("(PRM − Random) / (Oracle − Random)")
    ax.set_title("BoN@8 Normalized Lift", fontweight="bold")
    ax.legend(fontsize=5.5)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Cluster-Path PRM: Cross-Task Comparison", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "prm_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ─── 2. BoN curves grid: all tasks, all N values ─────────────────
    ncols = min(5, n_tasks)
    nrows = (n_tasks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for idx, t in enumerate(tasks):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sr = all_results[t][levels[0]]["sr"]

        for m in methods:
            bon = all_results[t][m]["strategies"]["final"]["bon"]
            ns = sorted(int(k) for k in bon.keys())
            means = np.array([(bon.get(n) or bon[str(n)])["mean"] for n in ns])
            ax.plot(ns, means, "o-", color=LEVEL_COLORS.get(m, "gray"),
                    label=LEVEL_LABELS.get(m, m), ms=4, lw=1.5)

        oracle = [1 - (1 - sr) ** N for N in ns]
        ax.plot(ns, oracle, "s--", color="black", alpha=0.4, ms=3, lw=1, label="Oracle")
        ax.axhline(sr, ls=":", color="gray", alpha=0.5, lw=0.8)

        ax.set_title(f"{_short_label(t)} (SR={sr:.0%})", fontsize=9, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns], fontsize=7)
        ax.set_ylim(max(0, sr - 0.05), min(1.05, max(oracle) + 0.08))
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    axes[0][0].legend(fontsize=6, loc="lower right")
    for idx in range(n_tasks, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.supxlabel("N (Best-of-N)", fontsize=10)
    fig.supylabel("Pass rate", fontsize=10)
    fig.suptitle("Best-of-N Curves Across Tasks", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "bon_curves_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ─── 3. Normalized lift over random: (PRM - random) / (oracle - random) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, lvl in enumerate(levels):
        lifts = []
        for t in tasks:
            bon8 = _get_bon(all_results[t][lvl], 8)
            prm_val = bon8["mean"]
            rnd = bon8["random"]
            orc = bon8["oracle"]
            gap = orc - rnd
            lifts.append((prm_val - rnd) / gap if gap > 0.01 else 0.0)
        ax.bar(x + i * width, lifts, width, label=LEVEL_LABELS[lvl], color=LEVEL_COLORS[lvl])

    ax.axhline(0, ls="-", color="gray", alpha=0.3, lw=0.8)
    ax.axhline(1, ls="--", color="black", alpha=0.3, lw=0.8, label="Oracle")
    ax.set_xticks(x + width)
    ax.set_xticklabels(short, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("Normalized lift  (PRM − Random) / (Oracle − Random)")
    ax.set_title("BoN@8 Normalized Lift Over Random", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "bon8_normalized_lift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ─── 4. Strategy comparison heatmap ──────────────────────────────
    strats = ["final", "min", "mean"]
    n_levels = len(levels)
    fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 4.5))
    if n_levels == 1:
        axes = [axes]
    for li, lvl in enumerate(levels):
        ax = axes[li]
        data = np.zeros((len(strats), n_tasks))
        for si, strat in enumerate(strats):
            for ti, t in enumerate(tasks):
                bon = all_results[t][lvl]["strategies"][strat]["bon"]
                key = 8 if 8 in bon else "8"
                data[si, ti] = bon[key]["mean"]
        im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(n_tasks))
        ax.set_xticklabels(short, fontsize=6, rotation=40, ha="right")
        ax.set_yticks(range(len(strats)))
        ax.set_yticklabels(strats, fontsize=8)
        ax.set_title(LEVEL_LABELS[lvl], fontweight="bold")
        for si in range(len(strats)):
            for ti in range(n_tasks):
                v = data[si, ti]
                ax.text(ti, si, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.65 else "black")

    fig.colorbar(im, ax=axes, shrink=0.8, label="BoN@8 pass rate")
    fig.suptitle("BoN@8 by Aggregation Strategy", fontsize=12, fontweight="bold")
    fig.savefig(out_dir / "strategy_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ─── 5. Summary table as figure ──────────────────────────────────
    method_short = {"level0_markov": "L0", "level1_likelihood": "L1",
                    "level2_logreg": "L2", "causal_emb_sim": "CosC",
                    "causal_knn": "CkNN"}
    col_labels = ["Task", "SR", "K", "n"]
    for m in methods:
        ms = method_short.get(m, m[:4])
        col_labels += [f"{ms} BoN@8"]
    col_labels += ["Oracle"]
    for m in methods:
        ms = method_short.get(m, m[:4])
        col_labels += [f"{ms} AUROC"]

    rows = []
    for t in tasks:
        first_r = all_results[t][methods[0]]
        sr = first_r["sr"]
        orc = _get_bon(first_r, 8)["oracle"]
        row = [_short_label(t), f"{sr:.0%}", str(first_r["K"]), str(first_r["n_rollouts"])]
        for m in methods:
            row.append(f"{_get_bon(all_results[t][m], 8)['mean']:.3f}")
        row.append(f"{orc:.3f}")
        for m in methods:
            row.append(f"{all_results[t][m]['strategies']['final']['auroc_mean']:.3f}")
        rows.append(row)

    fig, ax = plt.subplots(figsize=(16, 0.4 * n_tasks + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.35)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f4fa")

    ax.set_title("PRM Cross-Task Summary", fontsize=12, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved dashboard, BoN grid, normalized lift, strategy heatmap, summary table")


# ── Main runner ──────────────────────────────────────────────────────

LEVELS = {
    "level0_markov": (MarkovAbsorptionPRM, {"smoothing": 1.0}),
    "level1_likelihood": (LikelihoodRatioPRM, {"smoothing": 1.0}),
    "level2_logreg": (LogisticRegressionPRM, {"C": 1.0}),
    "level3_fasttext": (FastTextPRM, {"wordNgrams": 5, "dim": 20}),
    "level4_gru": (GRUPRM, {"embed_dim": 8, "hidden_dim": 16, "epochs": 200}),
}


def _build_levels_for_task(
    task: TaskData,
    level_filter: set[str] | None = None,
) -> dict:
    """Build PRM levels + embedding baselines for a task.

    If *level_filter* is given, only include levels whose key is in the set.
    """
    levels = dict(LEVELS)
    if task.step_embeddings is not None and task.rid_to_emb_rows is not None:
        emb_args = {
            "step_embeddings": task.step_embeddings,
            "rid_to_rows": task.rid_to_emb_rows,
        }
        levels["causal_emb_sim"] = (CausalEmbSimBaseline, dict(emb_args))
        levels["causal_knn"] = (CausalKnnBaseline, {**emb_args, "k": 5})
    if level_filter:
        levels = {k: v for k, v in levels.items() if k in level_filter}
    return levels


def run_all(
    data_dir: str,
    out_dir: str,
    action_embed_dir: str | None = None,
    response_embed_dir: str | None = None,
    min_cluster_size: int = 5,
    noise_strategy: str = "nearest",
    tool_arg_beta: float = 0.0,
    level_filter: set[str] | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CLUSTER SEQUENCE PRM EVALUATION")
    print(f"  noise_strategy = {noise_strategy!r}")
    if tool_arg_beta > 0:
        print(f"  tool_arg_beta  = {tool_arg_beta}")
    print("=" * 70)

    tasks = load_all_tasks(
        data_dir, action_embed_dir, response_embed_dir, min_cluster_size,
        noise_strategy=noise_strategy,
        tool_arg_beta=tool_arg_beta,
    )

    all_results: dict[str, dict[str, dict]] = {}

    for task in tasks:
        sr = float(np.mean(task.outcome))
        print(f"\n{'#' * 70}")
        print(f"# {task.label}  (K={task.K}, SR={sr:.0%}, n={len(task.all_rids)})")
        print(f"{'#' * 70}")

        task_dir = out / task.label.replace("/", "_")
        task_dir.mkdir(parents=True, exist_ok=True)
        task_results = {}

        task_levels = _build_levels_for_task(task, level_filter=level_filter)
        for level_name, (prm_class, prm_kwargs) in task_levels.items():
            print(f"\n  {level_name}...")
            result = run_cv_for_task(task, prm_class, prm_kwargs)

            with open(task_dir / f"{level_name}.json", "w") as f:
                json.dump(result, f, indent=2)

            task_results[level_name] = result

            bon8 = result["strategies"]["final"]["bon"][8]["mean"]
            auroc = result["strategies"]["final"]["auroc_mean"]
            print(f"    final: BoN@8={bon8:.3f}  AUROC={auroc:.3f}")

        plot_bon_per_task(task.label, task_results, task_dir / "bon_curves.png")
        all_results[task.label] = task_results

    # ── cross-task summary ──
    print(f"\n{'=' * 70}")
    print("CROSS-TASK SUMMARY")
    print(f"{'=' * 70}")

    with open(out / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    header = f"{'Task':<30s} {'Method':<20s} {'SR':>5s} {'BoN@8':>7s} {'Oracle':>7s} {'AUROC':>7s}"
    print(f"\n{header}")
    print("-" * len(header))
    for tl in sorted(all_results.keys()):
        for lname in ALL_METHODS:
            if lname not in all_results[tl]:
                continue
            r = all_results[tl][lname]
            bon8 = r["strategies"]["final"]["bon"][8]["mean"]
            orc = r["strategies"]["final"]["bon"][8]["oracle"]
            auroc = r["strategies"]["final"]["auroc_mean"]
            print(
                f"{tl:<30s} {LEVEL_LABELS.get(lname, lname):<20s} "
                f"{r['sr']:>5.0%} {bon8:>7.3f} {orc:>7.3f} {auroc:>7.3f}"
            )

    plot_cross_task_summary(all_results, out)
    print(f"\nResults saved to {out}")
    return all_results


# ── CLI ──────────────────────────────────────────────────────────────


def analyze_complexity(
    data_dir: str, out_dir: str, min_cluster_size: int = 5,
    noise_strategy: str = "nearest",
):
    """Correlate PRM performance with task complexity (power law)."""
    from scipy.optimize import curve_fit

    out = Path(out_dir)
    with open(out / "all_results.json") as f:
        all_results = json.load(f)

    tasks = load_all_tasks(data_dir, min_cluster_size=min_cluster_size,
                           noise_strategy=noise_strategy)
    task_map = {t.label: t for t in tasks}

    rows = []
    for task_label, level_results in all_results.items():
        td = task_map.get(task_label)
        if td is None:
            continue
        sr = td.outcome.mean()
        n = len(td.all_rids)
        n_pass = int(td.outcome.sum())
        n_fail = n - n_pass
        n_minority = min(n_pass, n_fail)
        K = td.K
        mean_len = float(np.mean([len(p) for p in td.rollout_paths.values()]))

        complexity = np.log(max(n_minority, 1)) / np.log(max(K * mean_len, 2))

        for lvl in LEVEL_ORDER:
            r = level_results[lvl]
            bon8 = _get_bon(r, 8)
            lift = (bon8["mean"] - bon8["random"]) / max(bon8["oracle"] - bon8["random"], 0.01)
            auroc = r["strategies"]["final"]["auroc_mean"]
            rows.append({
                "task": task_label,
                "short": _short_label(task_label),
                "level": lvl,
                "sr": sr,
                "n_minority": n_minority,
                "K": K,
                "mean_len": mean_len,
                "complexity": complexity,
                "bon8": bon8["mean"],
                "lift": lift,
                "auroc": auroc,
            })

    # ── Plot: complexity vs lift, one panel per level ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    def _sigmoid(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    for i, lvl in enumerate(LEVEL_ORDER):
        ax = axes[i]
        subset = [r for r in rows if r["level"] == lvl]
        cx = np.array([r["complexity"] for r in subset])
        lift = np.array([r["lift"] for r in subset])

        ax.scatter(cx, lift, c=LEVEL_COLORS[lvl], s=60, zorder=5, edgecolors="white", lw=0.5)
        for r in subset:
            ax.annotate(r["short"], (r["complexity"], r["lift"]),
                        fontsize=6, ha="center", va="bottom", xytext=(0, 4),
                        textcoords="offset points")

        try:
            popt, _ = curve_fit(_sigmoid, cx, lift, p0=[1.0, 5.0, 0.5],
                                bounds=([0, 0, -2], [1.5, 50, 3]), maxfev=5000)
            xs = np.linspace(cx.min() - 0.1, cx.max() + 0.1, 100)
            ax.plot(xs, _sigmoid(xs, *popt), "--", color=LEVEL_COLORS[lvl], alpha=0.6)
        except Exception:
            pass

        corr = float(np.corrcoef(cx, lift)[0, 1]) if len(cx) > 2 else 0
        ax.set_title(f"{LEVEL_LABELS[lvl]}  (r={corr:.2f})", fontweight="bold")
        ax.set_xlabel("log(n_minority) / log(K × mean_len)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Normalized Lift  (PRM−Rnd)/(Orc−Rnd)")
    fig.suptitle("PRM Performance vs Task Complexity", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "complexity_vs_lift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot: complexity vs AUROC ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for i, lvl in enumerate(LEVEL_ORDER):
        ax = axes[i]
        subset = [r for r in rows if r["level"] == lvl]
        cx = np.array([r["complexity"] for r in subset])
        auroc = np.array([r["auroc"] for r in subset])

        ax.scatter(cx, auroc, c=LEVEL_COLORS[lvl], s=60, zorder=5, edgecolors="white", lw=0.5)
        for r in subset:
            ax.annotate(r["short"], (r["complexity"], r["auroc"]),
                        fontsize=6, ha="center", va="bottom", xytext=(0, 4),
                        textcoords="offset points")

        try:
            popt, _ = curve_fit(_sigmoid, cx, auroc, p0=[1.0, 10.0, 0.3],
                                bounds=([0.5, 0, -2], [1.1, 50, 3]), maxfev=5000)
            xs = np.linspace(cx.min() - 0.1, cx.max() + 0.1, 100)
            ax.plot(xs, _sigmoid(xs, *popt), "--", color=LEVEL_COLORS[lvl], alpha=0.6)
        except Exception:
            pass

        corr = float(np.corrcoef(cx, auroc)[0, 1]) if len(cx) > 2 else 0
        ax.set_title(f"{LEVEL_LABELS[lvl]}  (r={corr:.2f})", fontweight="bold")
        ax.set_xlabel("log(n_minority) / log(K × mean_len)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("AUROC")
    fig.suptitle("AUROC vs Task Complexity", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "complexity_vs_auroc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Print summary ──
    print(f"\n{'Task':<20s} {'SR':>5s} {'nMin':>5s} {'K':>4s} {'mLen':>5s} {'Cmplx':>6s} "
          f"{'L0 Lift':>8s} {'L1 Lift':>8s} {'L2 Lift':>8s}")
    print("-" * 85)
    seen = set()
    for r in rows:
        if r["task"] in seen:
            continue
        seen.add(r["task"])
        l0 = next(x for x in rows if x["task"] == r["task"] and x["level"] == "level0_markov")
        l1 = next(x for x in rows if x["task"] == r["task"] and x["level"] == "level1_likelihood")
        l2 = next(x for x in rows if x["task"] == r["task"] and x["level"] == "level2_logreg")
        print(f"{r['short']:<20s} {r['sr']:>5.0%} {r['n_minority']:>5d} {r['K']:>4d} "
              f"{r['mean_len']:>5.1f} {r['complexity']:>6.3f} "
              f"{l0['lift']:>8.2f} {l1['lift']:>8.2f} {l2['lift']:>8.2f}")

    print(f"\nSaved complexity_vs_lift.png and complexity_vs_auroc.png to {out}")


def replot(out_dir: str):
    """Regenerate plots from saved all_results.json without re-running CV."""
    out = Path(out_dir)
    with open(out / "all_results.json") as f:
        all_results = json.load(f)

    for task_label, level_results in all_results.items():
        task_dir = out / task_label.replace("/", "_")
        task_dir.mkdir(parents=True, exist_ok=True)
        plot_bon_per_task(task_label, level_results, task_dir / "bon_curves.png")

    plot_cross_task_summary(all_results, out)
    print(f"Plots regenerated in {out}")


# ── Step-level Best-of-N ─────────────────────────────────────────────

STEP_BON_N_VALUES = [2, 4, 8, 16]


def _get_oos_step_scores(
    task: TaskData,
    prm_class,
    prm_kwargs: dict,
    seed: int = 42,
) -> dict[int, list[float]]:
    """5-fold CV to get out-of-sample per-step scores for every rollout."""
    rids = np.array(task.all_rids)
    rid_scores: dict[int, list[float]] = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(rids):
        train_rids = rids[train_idx].tolist()
        test_rids_fold = rids[test_idx].tolist()

        prm = prm_class(**prm_kwargs)
        prm.fit(
            {r: task.rollout_paths[r] for r in train_rids},
            {r: task.rewards[r] for r in train_rids},
            task.K,
        )

        for r in test_rids_fold:
            rid_scores[r] = prm.score_path(task.rollout_paths[r], rid=r)

    return rid_scores


def compute_step_level_bon(
    tasks: list[TaskData],
    out_dir: str,
    n_values: list[int] | None = None,
    n_draws: int = 2000,
) -> dict:
    """Step-level BoN: at each step T, PRM picks the best of N active rollouts."""
    if n_values is None:
        n_values = STEP_BON_N_VALUES
    out = Path(out_dir) / "step_bon"
    out.mkdir(parents=True, exist_ok=True)

    all_step_bon: dict[str, dict] = {}

    for task in tasks:
        sr = float(task.outcome.mean())
        print(f"\n{'=' * 60}")
        print(f"Step-level BoN: {task.label}  (K={task.K}, SR={sr:.0%}, n={len(task.all_rids)})")
        print(f"{'=' * 60}")

        outcomes = {r: float(task.rewards.get(r, 0) >= 0.5) for r in task.all_rids}

        task_levels = _build_levels_for_task(task)
        level_scores: dict[str, dict[int, list[float]]] = {}
        for lname, (cls, kwargs) in task_levels.items():
            print(f"  Scoring {lname}...", end="", flush=True)
            level_scores[lname] = _get_oos_step_scores(task, cls, kwargs)
            print(" done")

        max_steps = max(len(task.rollout_paths[r]) for r in task.all_rids)

        step_results: dict[int, dict] = {}
        for T in range(max_steps):
            active = [r for r in task.all_rids if len(task.rollout_paths[r]) > T]
            if len(active) < 2:
                continue

            sr_T = float(np.mean([outcomes[r] for r in active]))
            entry: dict = {
                "step": T,
                "n_active": len(active),
                "sr_at_step": sr_T,
                "oracle": {N: 1.0 - (1.0 - sr_T) ** N for N in n_values},
                "levels": {},
            }

            rng = np.random.default_rng(T * 10_000 + 42)

            for lname, rid_scores in level_scores.items():
                scored = [
                    r for r in active
                    if r in rid_scores and len(rid_scores[r]) > T
                ]
                if len(scored) < 2:
                    continue

                rids_arr = np.array(scored)
                out_arr = np.array([outcomes[r] for r in scored])
                sc_arr = np.array([rid_scores[r][T] for r in scored])
                M = len(rids_arr)

                level_bon: dict[int, float] = {}
                for N in n_values:
                    eff = min(N, M)
                    idx = rng.integers(0, M, size=(n_draws, eff))
                    best = idx[np.arange(n_draws), sc_arr[idx].argmax(axis=1)]
                    level_bon[N] = float(out_arr[best].mean())

                entry["levels"][lname] = level_bon

            step_results[T] = entry

        all_step_bon[task.label] = step_results

        for lname in level_scores:
            vals = [
                step_results[T]["levels"].get(lname, {}).get(8, None)
                for T in sorted(step_results)
            ]
            vals = [v for v in vals if v is not None]
            if vals:
                print(
                    f"  {LEVEL_LABELS.get(lname, lname):15s}  "
                    f"mean step-BoN@8 = {np.mean(vals):.3f}  "
                    f"(range {min(vals):.3f}\u2013{max(vals):.3f})"
                )

    json_out = {
        tl: {str(k): v for k, v in sd.items()}
        for tl, sd in all_step_bon.items()
    }
    with open(out / "step_bon_results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    plot_step_level_bon(all_step_bon, out, n_values)
    return all_step_bon


# ── Disagreement analysis ────────────────────────────────────────────


def analyze_disagreements(
    tasks: list,
    out_dir: str,
    primary_n: int = 8,
    n_draws: int = 5000,
    min_active: int = 30,
):
    """Identify steps where PRM selection disagrees with actual outcomes.

    A *disagreement* at step T occurs when the PRM's top pick from N sampled
    active rollouts *fails*, yet at least one sampled rollout *passes*.

    For each such event we record the score-gap: how much higher the PRM
    scored its (failing) pick vs the best passing alternative.
    """
    import math as _math

    out = Path(out_dir) / "disagreements"
    out.mkdir(parents=True, exist_ok=True)

    focus_levels = ["level1_likelihood", "level2_logreg"]

    task_summaries: dict[str, dict] = {}

    for task in tasks:
        sr = float(task.outcome.mean())
        print(f"\nDisagreement analysis: {task.label}  (SR={sr:.0%})")

        outcomes = {r: float(task.rewards.get(r, 0) >= 0.5) for r in task.all_rids}

        task_levels = _build_levels_for_task(task)
        level_scores: dict[str, dict[int, list[float]]] = {}
        for lname in focus_levels:
            if lname not in task_levels:
                continue
            cls, kwargs = task_levels[lname]
            level_scores[lname] = _get_oos_step_scores(task, cls, kwargs)

        max_steps = max(len(task.rollout_paths[r]) for r in task.all_rids)

        per_level: dict[str, dict] = {}
        for lname, rid_scores in level_scores.items():
            step_disagree_rate: dict[int, float] = {}
            step_magnitudes: dict[int, list[float]] = {}
            all_magnitudes: list[float] = []
            all_steps_with_errors: list[int] = []

            for T in range(max_steps):
                active = [r for r in task.all_rids if len(task.rollout_paths[r]) > T]
                if len(active) < min_active:
                    continue
                scored = [r for r in active
                          if r in rid_scores and len(rid_scores[r]) > T]
                if len(scored) < primary_n:
                    continue

                rids_arr = np.array(scored)
                out_arr = np.array([outcomes[r] for r in scored])
                sc_arr = np.array([rid_scores[r][T] for r in scored])
                M = len(rids_arr)

                rng = np.random.default_rng(T * 10_000 + 99)
                eff = min(primary_n, M)
                idx = rng.integers(0, M, size=(n_draws, eff))

                pick_idx = idx[np.arange(n_draws), sc_arr[idx].argmax(axis=1)]
                pick_outcome = out_arr[pick_idx]
                pick_score = sc_arr[pick_idx]

                sample_outcomes = out_arr[idx]
                any_pass = sample_outcomes.max(axis=1) > 0.5

                disagree = (pick_outcome < 0.5) & any_pass
                n_disagree = int(disagree.sum())
                n_eligible = int(any_pass.sum())

                if n_eligible > 0:
                    step_disagree_rate[T] = n_disagree / n_eligible

                if n_disagree > 0:
                    sample_scores = sc_arr[idx]
                    sample_pass_scores = np.where(sample_outcomes > 0.5, sample_scores, -np.inf)
                    best_pass_score = sample_pass_scores.max(axis=1)

                    gaps = pick_score[disagree] - best_pass_score[disagree]
                    step_magnitudes[T] = gaps.tolist()
                    all_magnitudes.extend(gaps.tolist())
                    all_steps_with_errors.extend([T] * n_disagree)

            per_level[lname] = {
                "step_disagree_rate": step_disagree_rate,
                "step_magnitudes": step_magnitudes,
                "all_magnitudes": all_magnitudes,
                "error_steps": all_steps_with_errors,
                "mean_disagree_rate": (
                    float(np.mean(list(step_disagree_rate.values())))
                    if step_disagree_rate else 0
                ),
            }

            n_steps = len(step_disagree_rate)
            mean_rate = per_level[lname]["mean_disagree_rate"]
            n_errs = len(all_magnitudes)
            print(f"  {LEVEL_LABELS.get(lname, lname):15s}  "
                  f"steps={n_steps}  mean_disagree_rate={mean_rate:.3f}  "
                  f"total_errors={n_errs}  "
                  f"median_gap={np.median(all_magnitudes):.4f}" if all_magnitudes else
                  f"  {LEVEL_LABELS.get(lname, lname):15s}  steps={n_steps}  no errors")

        task_summaries[task.label] = per_level

    _plot_disagreements(task_summaries, out, min_active)

    prm_root = Path(out_dir)
    for task in tasks:
        tl = task.label
        if tl not in task_summaries:
            continue
        task_folder = prm_root / tl.replace("/", "_")
        task_folder.mkdir(parents=True, exist_ok=True)
        for lname in focus_levels:
            if lname not in task_summaries[tl]:
                continue
            _plot_ridgeline(
                task_summaries[tl][lname]["step_magnitudes"],
                lname, tl, task_folder,
            )

    print(f"\nDisagreement plots saved to {out}")
    return task_summaries


def _plot_ridgeline(
    step_magnitudes: dict[int, list[float]],
    level_name: str,
    task_label: str,
    out_dir: Path,
):
    """Ridgeline plot: one KDE row per step showing error magnitude distribution."""
    from scipy.stats import gaussian_kde

    steps = sorted(s for s, v in step_magnitudes.items() if len(v) >= 10)
    if not steps:
        return

    color = LEVEL_COLORS.get(level_name, "#555555")
    label = LEVEL_LABELS.get(level_name, level_name)

    n_steps = len(steps)
    fig_h = max(4, 0.6 * n_steps + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    all_gaps = np.concatenate([step_magnitudes[s] for s in steps])
    lo, hi = np.percentile(all_gaps, [1, 99])
    pad = (hi - lo) * 0.1
    x_grid = np.linspace(lo - pad, hi + pad, 300)

    spacing = 1.0
    max_density = 0
    for s in steps:
        vals = np.array(step_magnitudes[s])
        try:
            kde = gaussian_kde(vals, bw_method=0.3)
            density = kde(x_grid)
            max_density = max(max_density, density.max())
        except Exception:
            pass

    scale = spacing * 0.85 / max(max_density, 1e-10)

    for i, s in enumerate(steps):
        vals = np.array(step_magnitudes[s])
        baseline = i * spacing
        try:
            kde = gaussian_kde(vals, bw_method=0.3)
            density = kde(x_grid) * scale
        except Exception:
            continue

        ax.fill_between(x_grid, baseline, baseline + density,
                         alpha=0.55, color=color, edgecolor="white", lw=0.5)
        ax.plot(x_grid, baseline + density, color=color, lw=0.8, alpha=0.8)

        med = np.median(vals)
        ax.plot(med, baseline, "|", color="black", ms=6, mew=1.2)

    ax.set_yticks([i * spacing for i in range(n_steps)])
    ax.set_yticklabels([f"Step {s}  (n={len(step_magnitudes[s])})" for s in steps],
                       fontsize=7)
    ax.set_xlabel("Score gap (PRM pick − best passing alternative)")
    ax.axvline(0, ls=":", color="gray", alpha=0.4)
    ax.set_title(f"{_short_label(task_label)} — {label}\nError Magnitude by Step",
                 fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.15, axis="x")
    ax.set_ylim(-0.5, n_steps * spacing + 0.3)

    fig.tight_layout()
    fname = f"disagree_ridgeline_{level_name}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / fname}")


def build_branching_manifest(
    tasks: list,
    out_dir: str,
    top_k_tasks: int = 3,
    n_targets: int = 50,
    min_active: int = 30,
    level_name: str = "level1_likelihood",
):
    """Build a weighted-sampled manifest of branching targets from high-margin errors.

    For each of the top-k best-calibrated tasks (by step-BoN lift), we:
      1. Get OOS PRM scores for every rollout at every step
      2. At each step T, identify failing rollouts scored above the median
         passing rollout — these are "confident errors"
      3. Compute margin = failing_score - median_pass_score
      4. Weight-sample n_targets branching targets proportional to margin
    """
    out = Path(out_dir) / "branching"
    out.mkdir(parents=True, exist_ok=True)

    # ── Rank tasks by step-BoN lift ──
    task_lifts: list[tuple[float, object]] = []
    for task in tasks:
        sr = float(task.outcome.mean())
        task_levels = _build_levels_for_task(task)
        if level_name not in task_levels:
            continue
        cls, kwargs = task_levels[level_name]
        rid_scores = _get_oos_step_scores(task, cls, kwargs)

        max_steps = max(len(task.rollout_paths[r]) for r in task.all_rids)
        outcomes = {r: float(task.rewards.get(r, 0) >= 0.5) for r in task.all_rids}
        rng = np.random.default_rng(42)

        bon_vals = []
        sr_vals = []
        for T in range(max_steps):
            active = [r for r in task.all_rids if len(task.rollout_paths[r]) > T]
            if len(active) < min_active:
                continue
            scored = [r for r in active if r in rid_scores and len(rid_scores[r]) > T]
            if len(scored) < 8:
                continue
            sr_T = float(np.mean([outcomes[r] for r in scored]))
            sr_vals.append(sr_T)

            rids_arr = np.array(scored)
            out_arr = np.array([outcomes[r] for r in scored])
            sc_arr = np.array([rid_scores[r][T] for r in scored])
            M = len(rids_arr)
            idx = rng.integers(0, M, size=(2000, min(8, M)))
            best = idx[np.arange(2000), sc_arr[idx].argmax(axis=1)]
            bon_vals.append(float(out_arr[best].mean()))

        lift = np.mean(bon_vals) - np.mean(sr_vals) if bon_vals else 0
        task_lifts.append((lift, task, rid_scores, outcomes))

    task_lifts.sort(key=lambda x: -x[0])
    selected = task_lifts[:top_k_tasks]

    print(f"\nTop {top_k_tasks} tasks by step-BoN lift for branching:")
    for lift, task, _, _ in selected:
        print(f"  {task.label:<50s}  lift={lift:.3f}  SR={task.outcome.mean():.0%}")

    # ── Build candidate pool per task ──
    all_candidates: list[dict] = []

    for lift, task, rid_scores, outcomes in selected:
        max_steps = max(len(task.rollout_paths[r]) for r in task.all_rids)

        for T in range(max_steps):
            active = [r for r in task.all_rids if len(task.rollout_paths[r]) > T]
            if len(active) < min_active:
                continue
            scored = [r for r in active if r in rid_scores and len(rid_scores[r]) > T]
            if len(scored) < 8:
                continue

            pass_scores = [rid_scores[r][T] for r in scored if outcomes[r] >= 0.5]
            if not pass_scores:
                continue
            median_pass = float(np.median(pass_scores))

            for r in scored:
                if outcomes[r] >= 0.5:
                    continue
                s = rid_scores[r][T]
                margin = s - median_pass
                if margin <= 0:
                    continue
                all_candidates.append({
                    "task_label": task.label,
                    "domain": task.label.split("/")[0],
                    "rollout_id": int(r),
                    "step": int(T),
                    "prm_score": float(s),
                    "median_pass_score": float(median_pass),
                    "margin": float(margin),
                    "outcome": int(outcomes[r]),
                    "n_active": len(active),
                    "path_prefix": task.rollout_paths[r][:T + 1],
                })

    print(f"\nTotal candidate error events: {len(all_candidates)}")

    if not all_candidates:
        print("No candidates found.")
        return []

    # ── Weighted sampling proportional to margin ──
    margins = np.array([c["margin"] for c in all_candidates])
    weights = margins / margins.sum()

    rng = np.random.default_rng(123)
    chosen_idx = rng.choice(len(all_candidates), size=min(n_targets, len(all_candidates)),
                            replace=False, p=weights)
    manifest = [all_candidates[i] for i in chosen_idx]
    manifest.sort(key=lambda x: (-x["margin"]))

    # ── Save ──
    with open(out / "branching_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Summary ──
    print(f"\nBranching manifest: {len(manifest)} targets")
    print(f"{'Task':<30s} {'RID':>4s} {'Step':>4s} {'PRM':>6s} {'MedPass':>8s} "
          f"{'Margin':>7s}")
    print("-" * 65)
    for m in manifest[:20]:
        print(f"{_short_label(m['task_label']):<30s} {m['rollout_id']:>4d} "
              f"{m['step']:>4d} {m['prm_score']:>6.3f} "
              f"{m['median_pass_score']:>8.3f} {m['margin']:>7.4f}")
    if len(manifest) > 20:
        print(f"  ... ({len(manifest) - 20} more)")

    # ── Distribution plot ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.hist([m["margin"] for m in manifest], bins=25, color="#2ca02c",
            alpha=0.7, edgecolor="white")
    ax.set_xlabel("Margin (PRM score − median pass score)")
    ax.set_ylabel("Count")
    ax.set_title("Margin Distribution of Sampled Targets", fontweight="bold")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    task_counts = {}
    for m in manifest:
        tl = _short_label(m["task_label"])
        task_counts[tl] = task_counts.get(tl, 0) + 1
    ax.barh(list(task_counts.keys()), list(task_counts.values()),
            color="#1f78b4", alpha=0.7)
    ax.set_xlabel("Count")
    ax.set_title("Targets per Task", fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")

    ax = axes[2]
    ax.hist([m["step"] for m in manifest], bins=range(
        min(m["step"] for m in manifest),
        max(m["step"] for m in manifest) + 2),
        color="#ff7f0e", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    ax.set_title("Step Distribution of Targets", fontweight="bold")
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"Branching Manifest — {len(manifest)} targets, "
                 f"weighted by margin ({LEVEL_LABELS.get(level_name, level_name)})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out / "branching_manifest_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out / 'branching_manifest.json'}")
    print(f"Saved {out / 'branching_manifest_summary.png'}")

    return manifest


def _plot_disagreements(
    task_summaries: dict[str, dict],
    out_dir: Path,
    min_active: int = 30,
):
    import math as _math

    tasks = sorted(task_summaries.keys())
    n_tasks = len(tasks)
    focus = [l for l in ["level1_likelihood", "level2_logreg"]
             if any(l in task_summaries[t] for t in tasks)]
    if not focus:
        return

    # ── 1. Per-step disagreement rate grid ──
    cols = min(3, n_tasks)
    rows = _math.ceil(n_tasks / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, tl in enumerate(tasks):
        ax = axes[idx // cols][idx % cols]
        for lname in focus:
            if lname not in task_summaries[tl]:
                continue
            rates = task_summaries[tl][lname]["step_disagree_rate"]
            if not rates:
                continue
            steps = sorted(rates.keys())
            ax.plot(steps, [rates[T] for T in steps], "o-",
                    color=LEVEL_COLORS.get(lname, "gray"),
                    ms=3, lw=1.5,
                    label=LEVEL_LABELS.get(lname, lname))
        ax.set_title(_short_label(tl), fontweight="bold", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Disagreement rate")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n_tasks, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("PRM Selection Disagreement Rate per Step\n"
                 f"(BoN@8, ≥{min_active} active rollouts, rate = errors / eligible draws)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "disagree_rate_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'disagree_rate_grid.png'}")

    # ── 2. Magnitude histograms (all tasks combined, per level) ──
    fig, axes = plt.subplots(1, len(focus), figsize=(6 * len(focus), 4), squeeze=False)
    for i, lname in enumerate(focus):
        ax = axes[0][i]
        all_mags = []
        for tl in tasks:
            if lname in task_summaries[tl]:
                all_mags.extend(task_summaries[tl][lname]["all_magnitudes"])
        if all_mags:
            ax.hist(all_mags, bins=60, color=LEVEL_COLORS.get(lname, "gray"),
                    alpha=0.75, edgecolor="white", lw=0.5)
            ax.axvline(np.median(all_mags), ls="--", color="black", alpha=0.6, lw=1.2,
                       label=f"median = {np.median(all_mags):.4f}")
            ax.axvline(0, ls="-", color="gray", alpha=0.3)
            ax.legend(fontsize=8)
        ax.set_title(LEVEL_LABELS.get(lname, lname), fontweight="bold")
        ax.set_xlabel("Score gap (PRM pick − best passing alternative)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Disagreement Magnitude Distribution (all tasks pooled)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_dir / "disagree_magnitude_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'disagree_magnitude_hist.png'}")

    # ── 3. Per-task magnitude box plots ──
    fig, axes = plt.subplots(1, len(focus), figsize=(6 * len(focus), 5), squeeze=False)
    for i, lname in enumerate(focus):
        ax = axes[0][i]
        data = []
        labels = []
        for tl in tasks:
            if lname in task_summaries[tl] and task_summaries[tl][lname]["all_magnitudes"]:
                data.append(task_summaries[tl][lname]["all_magnitudes"])
                labels.append(_short_label(tl))
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(LEVEL_COLORS.get(lname, "gray"))
                patch.set_alpha(0.5)
            ax.axhline(0, ls="-", color="gray", alpha=0.3)
        ax.set_title(LEVEL_LABELS.get(lname, lname), fontweight="bold")
        ax.set_ylabel("Score gap")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Disagreement Magnitude by Task",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_dir / "disagree_magnitude_by_task.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'disagree_magnitude_by_task.png'}")

    # ── 4. Error density: one grid per level ──
    for lname in focus:
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
        for idx, tl in enumerate(tasks):
            ax = axes[idx // cols][idx % cols]
            if lname in task_summaries[tl]:
                err_steps = task_summaries[tl][lname]["error_steps"]
                if err_steps:
                    ax.hist(err_steps,
                            bins=range(min(err_steps), max(err_steps) + 2),
                            color=LEVEL_COLORS.get(lname, "gray"),
                            alpha=0.75, edgecolor="white", lw=0.5)
            ax.set_title(_short_label(tl), fontweight="bold", fontsize=10)
            ax.set_xlabel("Step")
            ax.set_ylabel("Error count")
            ax.grid(True, alpha=0.2, axis="y")

        for idx in range(n_tasks, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        short = LEVEL_LABELS.get(lname, lname)
        fig.suptitle(f"Selection Error Density — {short}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fname = f"disagree_error_density_{lname}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_dir / fname}")

    # ── 5. Summary table ──
    print("\n" + "=" * 80)
    print(f"{'Task':<22s} {'Level':<16s} {'Steps':>5s} {'MeanRate':>9s} "
          f"{'Errors':>7s} {'MedGap':>8s} {'P25Gap':>8s} {'P75Gap':>8s}")
    print("-" * 80)
    for tl in tasks:
        for lname in focus:
            if lname not in task_summaries[tl]:
                continue
            info = task_summaries[tl][lname]
            mags = info["all_magnitudes"]
            n_steps = len(info["step_disagree_rate"])
            mr = info["mean_disagree_rate"]
            if mags:
                med = np.median(mags)
                p25, p75 = np.percentile(mags, [25, 75])
            else:
                med = p25 = p75 = 0.0
            print(f"{_short_label(tl):<22s} "
                  f"{LEVEL_LABELS.get(lname, lname):<16s} "
                  f"{n_steps:>5d} {mr:>9.3f} {len(mags):>7d} "
                  f"{med:>8.4f} {p25:>8.4f} {p75:>8.4f}")


def plot_step_level_bon(
    all_step_bon: dict[str, dict[int, dict]],
    out_dir: Path,
    n_values: list[int],
    primary_n: int = 8,
    min_active: int = 30,
):
    """Generate step-level BoN visualizations."""
    import math as _math

    tasks = sorted(all_step_bon.keys())
    n_tasks = len(tasks)
    cols = min(3, n_tasks)
    rows = _math.ceil(n_tasks / cols)

    # ── Grid: step-level BoN@8 curves per task ──
    fig, axes = plt.subplots(
        rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False,
    )

    for idx, task_label in enumerate(tasks):
        ax = axes[idx // cols][idx % cols]
        step_data = all_step_bon[task_label]
        steps = [T for T in sorted(step_data.keys())
                 if step_data[T]["n_active"] >= min_active]

        if not steps:
            ax.set_title(_short_label(task_label), fontweight="bold", fontsize=10)
            ax.text(0.5, 0.5, f"<{min_active} active\nat all steps",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        sr_line = [step_data[T]["sr_at_step"] for T in steps]
        oracle_line = [step_data[T]["oracle"].get(primary_n, 0) for T in steps]
        n_active_line = [step_data[T]["n_active"] for T in steps]

        ax.fill_between(steps, sr_line, oracle_line, alpha=0.08, color="gray")
        ax.plot(steps, sr_line, "--", color="gray", alpha=0.6, lw=1.2, label="Random (SR)")
        ax.plot(
            steps, oracle_line, "--", color="black", alpha=0.4, lw=1.2,
            label=f"Oracle@{primary_n}",
        )

        for lname in ALL_METHODS:
            vals = []
            valid_steps = []
            for T in steps:
                v = step_data[T]["levels"].get(lname, {}).get(primary_n)
                if v is not None:
                    vals.append(v)
                    valid_steps.append(T)
            if vals:
                ax.plot(
                    valid_steps, vals, "o-",
                    color=LEVEL_COLORS.get(lname, "gray"),
                    ms=3, lw=1.5,
                    label=LEVEL_LABELS.get(lname, lname),
                )

        ax2 = ax.twinx()
        ax2.bar(steps, n_active_line, alpha=0.07, color="blue", width=0.8)
        ax2.set_ylabel("Active rollouts", fontsize=7, color="blue", alpha=0.5)
        ax2.tick_params(axis="y", labelsize=6, colors="blue")

        ax.set_title(_short_label(task_label), fontweight="bold", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel(f"BoN@{primary_n} pass rate")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.2)

        if idx == 0:
            ax.legend(fontsize=6, loc="lower right")

    for idx in range(n_tasks, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        f"Step-Level Best-of-{primary_n} Selection",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        out_dir / f"step_bon{primary_n}_grid.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  Saved {out_dir / f'step_bon{primary_n}_grid.png'}")

    # ── Summary bar chart: mean step-BoN@8 across tasks ──
    fig, ax = plt.subplots(figsize=(12, 5))
    short_labels = [_short_label(t) for t in tasks]
    x = np.arange(n_tasks)

    all_methods_present = []
    for m in ALL_METHODS:
        if any(
            any(step_data[T]["levels"].get(m) for T in step_data)
            for step_data in all_step_bon.values()
        ):
            all_methods_present.append(m)

    n_m = len(all_methods_present)
    w = 0.8 / max(n_m, 1)

    for i, m in enumerate(all_methods_present):
        means = []
        for tl in tasks:
            sd = all_step_bon[tl]
            vals = [sd[T]["levels"].get(m, {}).get(primary_n)
                    for T in sd if sd[T]["n_active"] >= min_active]
            vals = [v for v in vals if v is not None]
            means.append(np.mean(vals) if vals else 0)
        ax.bar(
            x + i * w, means, w,
            label=LEVEL_LABELS.get(m, m),
            color=LEVEL_COLORS.get(m, "gray"),
        )

    mean_sr = []
    mean_oracle = []
    for tl in tasks:
        sd = all_step_bon[tl]
        valid_T = [T for T in sd if sd[T]["n_active"] >= min_active]
        srs = [sd[T]["sr_at_step"] for T in valid_T]
        orcs = [sd[T]["oracle"].get(primary_n, 0) for T in valid_T]
        mean_sr.append(np.mean(srs) if srs else 0)
        mean_oracle.append(np.mean(orcs) if orcs else 0)

    mid = n_m * w / 2
    for xi, sr_val, orc in zip(x, mean_sr, mean_oracle):
        ax.plot(xi + mid, orc, "kD", ms=5, zorder=5)
        ax.plot(xi + mid, sr_val, "_", color="gray", ms=10, mew=2, zorder=5)
    ax.plot([], [], "kD", ms=5, label=f"Oracle@{primary_n}")
    ax.plot([], [], "_", color="gray", ms=8, mew=2, label="Random")

    ax.set_xticks(x + mid)
    ax.set_xticklabels(short_labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel(f"Mean step-level BoN@{primary_n}")
    ax.set_title(
        f"Step-Level BoN@{primary_n} \u2014 Averaged Across Steps",
        fontweight="bold",
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.08)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"step_bon{primary_n}_summary.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  Saved {out_dir / f'step_bon{primary_n}_summary.png'}")

    # ── Normalized lift over steps ──
    fig, axes = plt.subplots(
        rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False,
    )

    for idx, task_label in enumerate(tasks):
        ax = axes[idx // cols][idx % cols]
        step_data = all_step_bon[task_label]
        steps = [T for T in sorted(step_data.keys())
                 if step_data[T]["n_active"] >= min_active]

        for lname in LEVEL_ORDER:
            lift_vals = []
            lift_steps = []
            for T in steps:
                bon_val = step_data[T]["levels"].get(lname, {}).get(primary_n)
                sr_T = step_data[T]["sr_at_step"]
                orc_T = step_data[T]["oracle"].get(primary_n, 0)
                gap = orc_T - sr_T
                if bon_val is not None and gap > 0.01:
                    lift_vals.append((bon_val - sr_T) / gap)
                    lift_steps.append(T)
            if lift_vals:
                ax.plot(
                    lift_steps, lift_vals, "o-",
                    color=LEVEL_COLORS.get(lname, "gray"),
                    ms=3, lw=1.5,
                    label=LEVEL_LABELS.get(lname, lname),
                )

        ax.axhline(0, ls="--", color="gray", alpha=0.4)
        ax.axhline(1, ls="--", color="black", alpha=0.3)
        ax.set_title(_short_label(task_label), fontweight="bold", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Normalized Lift")
        ax.set_ylim(-0.1, 1.3)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=6, loc="lower right")

    for idx in range(n_tasks, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        f"Step-Level Normalized Lift (BoN@{primary_n})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        out_dir / f"step_bon{primary_n}_lift_grid.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  Saved {out_dir / f'step_bon{primary_n}_lift_grid.png'}")


def _assign_heldout_to_clusters(
    sub_act: np.ndarray,
    sub_res: np.ndarray,
    sub_labels: np.ndarray,
    heldout_act: np.ndarray,
    heldout_res: np.ndarray,
) -> np.ndarray:
    """Assign held-out steps to the nearest subsampled-step cluster.

    Uses the mean of action and response cosine similarities (matching
    the 'mean' paired_similarity method used for clustering).
    """
    eps = 1e-12
    sub_act_n = sub_act / np.maximum(np.linalg.norm(sub_act, axis=1, keepdims=True), eps)
    sub_res_n = sub_res / np.maximum(np.linalg.norm(sub_res, axis=1, keepdims=True), eps)
    ho_act_n = heldout_act / np.maximum(np.linalg.norm(heldout_act, axis=1, keepdims=True), eps)
    ho_res_n = heldout_res / np.maximum(np.linalg.norm(heldout_res, axis=1, keepdims=True), eps)

    S_cross = 0.5 * (ho_act_n @ sub_act_n.T) + 0.5 * (ho_res_n @ sub_res_n.T)
    nearest = np.argmax(S_cross, axis=1)
    return sub_labels[nearest]


# ── Subsample experiment ─────────────────────────────────────────────

SUBSAMPLE_SIZES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
SUBSAMPLE_REPEATS = 10


def run_subsample_experiment(
    data_dir: str,
    out_dir: str,
    min_cluster_size: int = 5,
    noise_strategy: str = "nearest",
    mcs_values: list[int] | None = None,
) -> dict:
    """Measure PRM degradation when clustering uses fewer rollouts.

    For each sample size, subsample N rollouts for clustering, then use
    nearest-neighbor to assign the remaining rollouts' steps. CV always
    evaluates on the full set of rollouts.

    Only runs cluster-dependent methods (L0-L3); causal embedding
    baselines are independent of clustering and skipped here.
    """
    import warnings
    from tqdm import tqdm

    warnings.filterwarnings("ignore", category=FutureWarning)

    if mcs_values is None:
        mcs_values = [min_cluster_size]

    out = Path(out_dir) / "subsample"
    out.mkdir(parents=True, exist_ok=True)

    base = Path(data_dir)
    action_dir = str(base / "embeddings" / "action")
    response_dir = str(base / "embeddings" / "response")

    task_meta = _discover_tasks(data_dir, action_dir)
    print(f"Subsample experiment: {len(task_meta)} tasks, "
          f"mcs_values={mcs_values}, sizes={SUBSAMPLE_SIZES}", flush=True)

    valid_sizes_per_task: list[int] = []
    task_raw: list[dict] = []
    for tmeta in task_meta:
        action_npz = tmeta["npz_path"]
        response_npz = str(Path(response_dir) / Path(action_npz).name)
        if not Path(response_npz).exists():
            continue
        act_mat, res_mat, indices = load_paired_embeddings(action_npz, response_npz)
        rewards = _load_rewards_from_files(tmeta["tdp_files"])
        all_rids_set = sorted(set(r for r, _ in indices) & set(rewards.keys()))
        n_total = len(all_rids_set)
        n_valid = sum(1 for s in SUBSAMPLE_SIZES if s <= n_total)
        valid_sizes_per_task.append(n_valid)
        task_raw.append({
            "meta": tmeta, "act_mat": act_mat, "res_mat": res_mat,
            "indices": indices, "rewards": rewards,
            "all_rids": all_rids_set, "n_total": n_total,
        })

    total_iters = sum(
        n_valid * SUBSAMPLE_REPEATS * len(mcs_values)
        for n_valid in valid_sizes_per_task
    )

    all_results: dict = {}
    pbar = tqdm(total=total_iters, desc="Subsample", unit="run")

    for tinfo in task_raw:
        tmeta = tinfo["meta"]
        label = tmeta["short_id"]
        act_mat, res_mat = tinfo["act_mat"], tinfo["res_mat"]
        indices, rewards = tinfo["indices"], tinfo["rewards"]
        all_rids_set = tinfo["all_rids"]
        n_total = tinfo["n_total"]

        pbar.set_postfix(task=label[:20])

        task_results: dict = {}

        for mcs in mcs_values:
            for n_sub in SUBSAMPLE_SIZES:
                if n_sub > n_total:
                    continue

                repeat_results: list[dict] = []

                for rep in range(SUBSAMPLE_REPEATS):
                    rng = np.random.default_rng(rep * 10_000 + n_sub * 100 + mcs)
                    sub_rids = sorted(rng.choice(all_rids_set, size=n_sub, replace=False))
                    sub_rid_set = set(sub_rids)

                    sub_mask = np.array([i for i, (r, _) in enumerate(indices) if r in sub_rid_set])
                    ho_mask = np.array([i for i, (r, _) in enumerate(indices) if r not in sub_rid_set])

                    sub_act, sub_res = act_mat[sub_mask], res_mat[sub_mask]
                    sub_indices = [indices[i] for i in sub_mask]

                    S_sub, _, _ = paired_similarity(sub_act, sub_res, "mean")
                    effective_mcs = max(2, min(mcs, n_sub // 5))
                    sub_labels = hdbscan_cluster_affinity(
                        S_sub, min_cluster_size=effective_mcs, noise_strategy=noise_strategy,
                    )
                    K = int(sub_labels.max()) + 1

                    if len(ho_mask) > 0:
                        ho_act, ho_res = act_mat[ho_mask], res_mat[ho_mask]
                        ho_indices = [indices[i] for i in ho_mask]
                        ho_labels = _assign_heldout_to_clusters(
                            sub_act, sub_res, sub_labels, ho_act, ho_res,
                        )
                        merged_labels = np.concatenate([sub_labels, ho_labels])
                        merged_indices = sub_indices + ho_indices
                    else:
                        merged_labels = sub_labels
                        merged_indices = sub_indices

                    _, rollout_paths = build_global_thought_graph(
                        merged_labels, merged_indices, rewards,
                    )
                    eval_rids = sorted(set(rollout_paths.keys()) & set(rewards.keys()))
                    rollout_paths = {r: rollout_paths[r] for r in eval_rids}
                    outcome = np.array(
                        [1.0 if rewards.get(r, 0) >= 0.5 else 0.0 for r in eval_rids]
                    )

                    td = TaskData(
                        label=label, domain=tmeta["domain"], task_id=tmeta["task_id"],
                        labels=merged_labels, indices=merged_indices,
                        rewards={r: rewards[r] for r in eval_rids},
                        rollout_paths=rollout_paths, K=K,
                        all_rids=eval_rids, outcome=outcome,
                        step_embeddings=None, rid_to_emb_rows=None,
                    )

                    rep_result: dict[str, dict] = {}
                    skip_levels = {"level3_fasttext"}
                    for level_name, (prm_class, prm_kwargs) in LEVELS.items():
                        if level_name in skip_levels:
                            continue
                        cv = run_cv_for_task(td, prm_class, prm_kwargs)
                        rep_result[level_name] = cv

                    repeat_results.append({"K": K, "levels": rep_result})
                    pbar.update(1)

                key = f"mcs{mcs}_n{n_sub}"
                task_results[key] = repeat_results

        all_results[label] = task_results

    pbar.close()

    results_path = out / "subsample_results.json"
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        for task_label, task_data in all_results.items():
            if task_label not in existing:
                existing[task_label] = {}
            existing[task_label].update(task_data)
        all_results = existing
        print("Merged with existing results", flush=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}", flush=True)

    plot_subsample_results(all_results, out, data_dir)
    return all_results


def run_subsample_hard_experiment(
    data_dir: str,
    out_dir: str,
    min_cluster_size: int = 5,
    noise_strategy: str = "nearest",
    mcs_values: list[int] | None = None,
) -> dict:
    """Hard subsample: cluster AND train PRM on N rollouts, evaluate on held-out.

    Unlike the soft variant which does CV on all 100 rollouts, here the PRM
    is trained exclusively on the N subsampled rollouts. Held-out rollouts are
    assigned to clusters via NN, then scored. AUROC and BoN are computed on
    the held-out set only.
    """
    import warnings
    from tqdm import tqdm

    warnings.filterwarnings("ignore", category=FutureWarning)

    if mcs_values is None:
        mcs_values = [min_cluster_size]

    out = Path(out_dir) / "subsample_hard"
    out.mkdir(parents=True, exist_ok=True)

    base = Path(data_dir)
    action_dir = str(base / "embeddings" / "action")
    response_dir = str(base / "embeddings" / "response")

    task_meta = _discover_tasks(data_dir, action_dir)
    print(f"Hard-subsample experiment: {len(task_meta)} tasks, "
          f"mcs_values={mcs_values}, sizes={SUBSAMPLE_SIZES}", flush=True)

    valid_sizes_per_task: list[int] = []
    task_raw: list[dict] = []
    for tmeta in task_meta:
        action_npz = tmeta["npz_path"]
        response_npz = str(Path(response_dir) / Path(action_npz).name)
        if not Path(response_npz).exists():
            continue
        act_mat, res_mat, indices = load_paired_embeddings(action_npz, response_npz)
        rewards = _load_rewards_from_files(tmeta["tdp_files"])
        all_rids_set = sorted(set(r for r, _ in indices) & set(rewards.keys()))
        n_total = len(all_rids_set)
        n_valid = sum(1 for s in SUBSAMPLE_SIZES if s <= n_total - 5)
        valid_sizes_per_task.append(n_valid)
        task_raw.append({
            "meta": tmeta, "act_mat": act_mat, "res_mat": res_mat,
            "indices": indices, "rewards": rewards,
            "all_rids": all_rids_set, "n_total": n_total,
        })

    total_iters = sum(
        n_valid * SUBSAMPLE_REPEATS * len(mcs_values)
        for n_valid in valid_sizes_per_task
    )

    all_results: dict = {}
    pbar = tqdm(total=total_iters, desc="Hard-subsample", unit="run")

    for tinfo in task_raw:
        tmeta = tinfo["meta"]
        label = tmeta["short_id"]
        act_mat, res_mat = tinfo["act_mat"], tinfo["res_mat"]
        indices, rewards = tinfo["indices"], tinfo["rewards"]
        all_rids_set = tinfo["all_rids"]
        n_total = tinfo["n_total"]

        pbar.set_postfix(task=label[:20])

        task_results: dict = {}

        for mcs in mcs_values:
            for n_sub in SUBSAMPLE_SIZES:
                if n_sub > n_total - 5:
                    continue

                repeat_results: list[dict] = []

                for rep in range(SUBSAMPLE_REPEATS):
                    rng = np.random.default_rng(rep * 10_000 + n_sub * 100 + mcs)
                    sub_rids = sorted(rng.choice(all_rids_set, size=n_sub, replace=False))
                    sub_rid_set = set(sub_rids)
                    ho_rid_set = set(all_rids_set) - sub_rid_set

                    sub_mask = np.array([i for i, (r, _) in enumerate(indices) if r in sub_rid_set])
                    ho_mask = np.array([i for i, (r, _) in enumerate(indices) if r in ho_rid_set])

                    sub_act, sub_res = act_mat[sub_mask], res_mat[sub_mask]
                    sub_indices = [indices[i] for i in sub_mask]

                    S_sub, _, _ = paired_similarity(sub_act, sub_res, "mean")
                    effective_mcs = max(2, min(mcs, n_sub // 5))
                    sub_labels = hdbscan_cluster_affinity(
                        S_sub, min_cluster_size=effective_mcs, noise_strategy=noise_strategy,
                    )
                    K = int(sub_labels.max()) + 1

                    # Build cluster paths for training rollouts only
                    _, sub_rollout_paths = build_global_thought_graph(
                        sub_labels, sub_indices, rewards,
                    )
                    train_rids = sorted(set(sub_rollout_paths.keys()) & set(rewards.keys()) & sub_rid_set)
                    sub_rollout_paths = {r: sub_rollout_paths[r] for r in train_rids}
                    train_rewards = {r: rewards[r] for r in train_rids}

                    # Assign held-out rollouts to clusters via NN
                    ho_act, ho_res = act_mat[ho_mask], res_mat[ho_mask]
                    ho_indices = [indices[i] for i in ho_mask]
                    ho_labels = _assign_heldout_to_clusters(
                        sub_act, sub_res, sub_labels, ho_act, ho_res,
                    )
                    _, ho_rollout_paths = build_global_thought_graph(
                        ho_labels, ho_indices, rewards,
                    )
                    ho_eval_rids = sorted(set(ho_rollout_paths.keys()) & set(rewards.keys()) & ho_rid_set)
                    ho_outcomes = {r: float(rewards.get(r, 0) >= 0.5) for r in ho_eval_rids}

                    rep_result: dict[str, dict] = {}
                    skip_levels = {"level3_fasttext"}
                    for level_name, (prm_class, prm_kwargs) in LEVELS.items():
                        if level_name in skip_levels:
                            continue

                        prm = prm_class(**prm_kwargs)
                        prm.fit(sub_rollout_paths, train_rewards, K)

                        step_scores: dict[int, list[float]] = {}
                        for r in ho_eval_rids:
                            if r in ho_rollout_paths:
                                step_scores[r] = prm.score_path(ho_rollout_paths[r], rid=r)

                        scored_rids = [r for r in ho_eval_rids if r in step_scores]
                        if len(scored_rids) < 5:
                            continue

                        sr = float(np.mean([ho_outcomes[r] for r in scored_rids]))
                        level_summary: dict = {
                            "task": label, "n_rollouts": len(scored_rids),
                            "sr": sr, "K": K,
                            "strategies": {},
                        }

                        rng_eval = np.random.default_rng(rep * 100_000 + n_sub * 1000 + 7)
                        for strat in STRATEGIES:
                            agg = {r: _aggregate_scores(step_scores[r], strat) for r in scored_rids}
                            bon = _evaluate_bon(scored_rids, ho_outcomes, agg, N_VALUES, rng_eval)
                            auroc_val = _evaluate_auroc(scored_rids, ho_outcomes, agg)
                            level_summary["strategies"][strat] = {
                                "auroc_mean": auroc_val,
                                "auroc_std": 0.0,
                                "bon": {
                                    N: {
                                        "mean": bon[N], "std": 0.0,
                                        "random": sr, "oracle": 1 - (1 - sr) ** N,
                                    }
                                    for N in N_VALUES
                                },
                            }

                        rep_result[level_name] = level_summary

                    repeat_results.append({"K": K, "levels": rep_result})
                    pbar.update(1)

                key = f"mcs{mcs}_n{n_sub}"
                task_results[key] = repeat_results

        all_results[label] = task_results

    pbar.close()

    results_path = out / "subsample_results.json"
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        for task_label, task_data in all_results.items():
            if task_label not in existing:
                existing[task_label] = {}
            existing[task_label].update(task_data)
        all_results = existing
        print("Merged with existing results", flush=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}", flush=True)

    plot_subsample_results(all_results, out, data_dir)
    return all_results


def plot_subsample_results(
    results: dict,
    out_dir: str | Path,
    data_dir: str | None = None,
):
    """Plot PRM performance vs subsample size with error bands."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    baseline_results: dict | None = None
    if data_dir:
        baseline_path = Path(data_dir) / "analysis" / "prm_noise_bin" / "all_results.json"
        if not baseline_path.exists():
            baseline_path = Path(data_dir) / "analysis" / "prm" / "all_results.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline_results = json.load(f)

    task_labels = sorted(results.keys())
    mcs_values = sorted(set(
        int(k.split("_")[0].replace("mcs", ""))
        for tl in results for k in results[tl]
    ))

    for mcs in mcs_values:
        for metric_name, metric_key, ylabel in [
            ("bon8", lambda cv: _get_bon(cv, 8)["mean"], "BoN@8 Accuracy"),
            ("auroc", lambda cv: cv["strategies"]["final"]["auroc_mean"], "AUROC"),
        ]:
            n_tasks = len(task_labels)
            fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4.5), squeeze=False)
            axes = axes[0]

            for ti, tl in enumerate(task_labels):
                ax = axes[ti]
                task_data = results[tl]

                sizes = sorted(set(
                    int(k.split("_")[1].replace("n", ""))
                    for k in task_data if k.startswith(f"mcs{mcs}_")
                ))

                for lv in LEVEL_ORDER:
                    means, stds = [], []
                    valid_sizes = []
                    for n_sub in sizes:
                        key = f"mcs{mcs}_n{n_sub}"
                        if key not in task_data:
                            continue
                        vals = []
                        for rr in task_data[key]:
                            if lv in rr["levels"]:
                                vals.append(metric_key(rr["levels"][lv]))
                        if vals:
                            means.append(np.mean(vals))
                            stds.append(np.std(vals))
                            valid_sizes.append(n_sub)

                    if not means:
                        continue

                    color = LEVEL_COLORS[lv]
                    lbl = LEVEL_LABELS[lv]
                    ax.plot(valid_sizes, means, "o-", color=color, label=lbl,
                            markersize=4, linewidth=1.5)
                    ax.fill_between(
                        valid_sizes,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=color, alpha=0.15,
                    )

                    if baseline_results and tl in baseline_results:
                        bl = baseline_results[tl]
                        if lv in bl:
                            bl_val = metric_key(bl[lv])
                            ax.axhline(bl_val, color=color, linestyle="--",
                                       alpha=0.5, linewidth=1)
                            ax.plot(100, bl_val, "s", color=color, markersize=6)

                if baseline_results and tl in baseline_results:
                    any_bl = next(iter(baseline_results[tl].values()))
                    sr = any_bl["sr"]
                else:
                    sr_key = f"mcs{mcs}_n{sizes[-1]}" if sizes else None
                    if sr_key and sr_key in task_data and task_data[sr_key]:
                        sr = task_data[sr_key][0]["levels"][
                            next(iter(task_data[sr_key][0]["levels"]))
                        ]["sr"]
                    else:
                        sr = 0.5

                if metric_name == "bon8":
                    oracle = 1 - (1 - sr) ** 8
                    ax.axhline(sr, color="gray", linestyle=":", alpha=0.5,
                               label="Random" if ti == 0 else None)
                    ax.axhline(oracle, color="black", linestyle=":", alpha=0.5,
                               label="Oracle" if ti == 0 else None)

                ax.set_xlabel("Rollouts used for clustering")
                if ti == 0:
                    ax.set_ylabel(ylabel)
                ax.set_title(f"{_short_label(tl)}\nsr = {sr:.2f}", fontsize=10)
                ax.grid(True, alpha=0.2)
                if ti == 0:
                    ax.legend(fontsize=7, loc="lower right")

            suptitle = f"Subsample: {ylabel} vs Clustering Rollouts"
            if len(mcs_values) > 1:
                suptitle += f" (mcs={mcs})"
            fig.suptitle(suptitle, fontsize=13, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.93])

            suffix = f"_mcs{mcs}" if len(mcs_values) > 1 else ""
            fig.savefig(out / f"subsample_{metric_name}{suffix}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    # K vs sample size plot
    fig, axes = plt.subplots(1, len(task_labels),
                             figsize=(5 * len(task_labels), 4), squeeze=False)
    axes = axes[0]
    for ti, tl in enumerate(task_labels):
        ax = axes[ti]
        task_data = results[tl]
        for mcs in mcs_values:
            sizes, k_means, k_stds = [], [], []
            matching_keys = [k for k in task_data if k.startswith(f"mcs{mcs}_")]
            matching_keys.sort(key=lambda k: int(k.split("_")[1].replace("n", "")))
            for key in matching_keys:
                n_sub = int(key.split("_")[1].replace("n", ""))
                ks = [rr["K"] for rr in task_data[key]]
                sizes.append(n_sub)
                k_means.append(np.mean(ks))
                k_stds.append(np.std(ks))
            if sizes:
                sizes_arr = np.array(sizes)
                k_means_arr = np.array(k_means)
                k_stds_arr = np.array(k_stds)
                eff_mcs = np.array([max(2, min(mcs, s // 5)) for s in sizes])
                adaptive_mask = eff_mcs < mcs
                nominal_mask = ~adaptive_mask

                lbl_nom = f"mcs={mcs}" if len(mcs_values) > 1 else "K"
                if nominal_mask.any():
                    ax.errorbar(sizes_arr[nominal_mask], k_means_arr[nominal_mask],
                                yerr=k_stds_arr[nominal_mask], fmt="o-",
                                capsize=3, color="#1f78b4", label=lbl_nom if ti == 0 else None)
                if adaptive_mask.any():
                    ax.errorbar(sizes_arr[adaptive_mask], k_means_arr[adaptive_mask],
                                yerr=k_stds_arr[adaptive_mask], fmt="s--",
                                capsize=3, color="#e31a1c",
                                label=f"adaptive mcs<{mcs}" if ti == 0 else None)
                # connect the two segments at the boundary
                if adaptive_mask.any() and nominal_mask.any():
                    bnd = adaptive_mask.sum()
                    ax.plot(sizes_arr[bnd - 1:bnd + 1], k_means_arr[bnd - 1:bnd + 1],
                            "--", color="gray", alpha=0.4, linewidth=1)
        ax.set_xlabel("Rollouts used for clustering")
        if ti == 0:
            ax.set_ylabel("Number of clusters (K)")
        if baseline_results and tl in baseline_results:
            any_bl = next(iter(baseline_results[tl].values()))
            sr_k = any_bl["sr"]
        else:
            first_key = next(iter(task_data), None)
            sr_k = task_data[first_key][0]["levels"][
                next(iter(task_data[first_key][0]["levels"]))
            ]["sr"] if first_key and task_data[first_key] else 0.5
        ax.set_title(f"{_short_label(tl)}\nsr = {sr_k:.2f}", fontsize=10)
        ax.grid(True, alpha=0.2)
        if ti == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Cluster Count vs Subsample Size", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / "subsample_K.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved subsample plots to {out}")


# ── Cluster convergence (ARI / NMI) ─────────────────────────────────


def run_subsample_convergence(
    data_dir: str,
    out_dir: str,
    min_cluster_size: int = 5,
    noise_strategy: str = "nearest",
) -> dict:
    """Measure cluster partition convergence via ARI and NMI.

    For each subsample size, cluster N rollouts' steps, NN-assign the rest,
    then compare the resulting partition to the reference (full 100-rollout)
    partition using Adjusted Rand Index and Normalized Mutual Information.
    """
    import warnings
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from tqdm import tqdm

    warnings.filterwarnings("ignore", category=FutureWarning)

    out = Path(out_dir) / "subsample"
    out.mkdir(parents=True, exist_ok=True)

    base = Path(data_dir)
    action_dir = str(base / "embeddings" / "action")
    response_dir = str(base / "embeddings" / "response")

    task_meta = _discover_tasks(data_dir, action_dir)
    print(f"Convergence experiment: {len(task_meta)} tasks, "
          f"mcs={min_cluster_size}, sizes={SUBSAMPLE_SIZES}", flush=True)

    task_raw: list[dict] = []
    for tmeta in task_meta:
        action_npz = tmeta["npz_path"]
        response_npz = str(Path(response_dir) / Path(action_npz).name)
        if not Path(response_npz).exists():
            continue
        act_mat, res_mat, indices = load_paired_embeddings(action_npz, response_npz)
        rewards = _load_rewards_from_files(tmeta["tdp_files"])
        all_rids_set = sorted(set(r for r, _ in indices) & set(rewards.keys()))
        n_total = len(all_rids_set)
        task_raw.append({
            "meta": tmeta, "act_mat": act_mat, "res_mat": res_mat,
            "indices": indices, "rewards": rewards,
            "all_rids": all_rids_set, "n_total": n_total,
        })

    total_iters = sum(
        sum(1 for s in SUBSAMPLE_SIZES if s <= t["n_total"]) * SUBSAMPLE_REPEATS
        for t in task_raw
    )

    all_results: dict = {}
    pbar = tqdm(total=total_iters, desc="Convergence", unit="run")

    for tinfo in task_raw:
        tmeta = tinfo["meta"]
        label = tmeta["short_id"]
        act_mat, res_mat = tinfo["act_mat"], tinfo["res_mat"]
        indices = tinfo["indices"]
        all_rids_set = tinfo["all_rids"]
        n_total = tinfo["n_total"]
        n_steps = len(indices)

        pbar.set_postfix(task=label[:20])

        # Reference partition: cluster all steps with full mcs
        S_full, _, _ = paired_similarity(act_mat, res_mat, "mean")
        ref_labels = hdbscan_cluster_affinity(
            S_full, min_cluster_size=min_cluster_size,
            noise_strategy=noise_strategy,
        )
        ref_K = int(ref_labels.max()) + 1
        print(f"  {label}: ref K={ref_K} ({n_steps} steps)", flush=True)

        # Build rid->step-index mapping for fast masking
        rid_set = set(all_rids_set)

        task_results: dict = {}

        for n_sub in SUBSAMPLE_SIZES:
            if n_sub > n_total:
                continue

            repeat_results: list[dict] = []

            for rep in range(SUBSAMPLE_REPEATS):
                rng = np.random.default_rng(rep * 10_000 + n_sub * 100 + min_cluster_size)
                sub_rids = sorted(rng.choice(all_rids_set, size=n_sub, replace=False))
                sub_rid_set = set(sub_rids)

                sub_mask = np.array([i for i, (r, _) in enumerate(indices) if r in sub_rid_set])
                ho_mask = np.array([i for i, (r, _) in enumerate(indices) if r not in sub_rid_set])

                sub_act, sub_res = act_mat[sub_mask], res_mat[sub_mask]

                S_sub, _, _ = paired_similarity(sub_act, sub_res, "mean")
                effective_mcs = max(2, min(min_cluster_size, n_sub // 5))
                sub_labels = hdbscan_cluster_affinity(
                    S_sub, min_cluster_size=effective_mcs,
                    noise_strategy=noise_strategy,
                )
                K = int(sub_labels.max()) + 1

                if len(ho_mask) > 0:
                    ho_act, ho_res = act_mat[ho_mask], res_mat[ho_mask]
                    ho_labels = _assign_heldout_to_clusters(
                        sub_act, sub_res, sub_labels, ho_act, ho_res,
                    )
                    merged_labels = np.concatenate([sub_labels, ho_labels])
                    merged_order = np.concatenate([sub_mask, ho_mask])
                else:
                    merged_labels = sub_labels
                    merged_order = sub_mask

                # Reorder merged labels to match original step ordering
                full_labels = np.empty(n_steps, dtype=merged_labels.dtype)
                full_labels[merged_order] = merged_labels

                ari = float(adjusted_rand_score(ref_labels, full_labels))
                nmi = float(normalized_mutual_info_score(ref_labels, full_labels))

                repeat_results.append({
                    "K": K, "ref_K": ref_K, "ari": ari, "nmi": nmi,
                })
                pbar.update(1)

            key = f"n{n_sub}"
            task_results[key] = repeat_results

        all_results[label] = {
            "ref_K": ref_K,
            "sr": tinfo["rewards"],
            "results": task_results,
        }

    pbar.close()

    # Store sr as a float
    for label in all_results:
        rewards = all_results[label]["sr"]
        if isinstance(rewards, dict):
            sr = sum(1 for r in rewards.values() if r >= 0.5) / len(rewards) if rewards else 0
            all_results[label]["sr"] = sr

    results_path = out / "subsample_convergence.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved convergence results to {results_path}", flush=True)

    plot_subsample_convergence(all_results, out)
    return all_results


def plot_subsample_convergence(
    results: dict,
    out_dir: str | Path,
):
    """Plot ARI and NMI vs subsample size with error bands."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    task_labels = sorted(results.keys())
    n_tasks = len(task_labels)

    fig, axes = plt.subplots(2, n_tasks, figsize=(5 * n_tasks, 8), squeeze=False)

    for ti, tl in enumerate(task_labels):
        task_data = results[tl]["results"]
        sr = results[tl].get("sr", 0.5)
        ref_K = results[tl].get("ref_K", "?")

        sizes = sorted(int(k.replace("n", "")) for k in task_data)

        for mi, (metric, metric_label, color) in enumerate([
            ("ari", "Adjusted Rand Index", "#1f78b4"),
            ("nmi", "Normalized Mutual Info", "#33a02c"),
        ]):
            ax = axes[mi][ti]
            means, stds = [], []
            for n_sub in sizes:
                key = f"n{n_sub}"
                vals = [r[metric] for r in task_data[key]]
                means.append(np.mean(vals))
                stds.append(np.std(vals))

            ax.plot(sizes, means, "o-", color=color, markersize=4, linewidth=1.5)
            ax.fill_between(
                sizes,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                color=color, alpha=0.15,
            )
            ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)
            ax.set_xlabel("Rollouts used for clustering")
            if ti == 0:
                ax.set_ylabel(metric_label)
            ax.set_title(f"{_short_label(tl)}\nsr={sr:.2f}, ref K={ref_K}", fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Cluster Partition Convergence vs Subsample Size",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / "subsample_ari_nmi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved convergence plot to {out / 'subsample_ari_nmi.png'}")


# ── Biased branch-point subsampling ──────────────────────────────────

BIASED_TARGET_TASKS = {"retail/17", "telecom hard", "telecom data"}

BIASED_STRATEGIES = ["minority_weighted", "minority_only", "branch_balanced"]

BIASED_PRIMARY_N = 80
BIASED_NESTED_SIZES = [30, 40, 50, 60, 70, 80]


def _match_target_task(short_id: str) -> bool:
    """Check if a task short_id matches one of the biased-subsample targets."""
    sl = _short_label(short_id)
    return sl in BIASED_TARGET_TASKS or short_id in BIASED_TARGET_TASKS


def compute_branch_weights(
    indices: list[tuple[int, int]],
    labels: np.ndarray,
    rewards: dict[int, float],
    rollout_paths: dict[int, list[int]],
    task_sr: float,
    strategy: str = "minority_weighted",
    early_steps: int = 3,
) -> dict[int, float]:
    """Compute per-rollout importance weights for biased subsampling.

    Strategies:
      minority_weighted  outcome_score * rarity_score (targets minority
                         outcome in small clusters)
      minority_only      outcome_score only (ablation)
      branch_balanced    rarity_score only (structural rarity, no outcome)
    """
    from collections import defaultdict

    step_cluster_rids: dict[tuple[int, int], set[int]] = defaultdict(set)
    for (rid, step), lbl in zip(indices, labels):
        if step < early_steps:
            step_cluster_rids[(step, int(lbl))].add(rid)

    step_cluster_sr: dict[tuple[int, int], float] = {}
    step_cluster_size: dict[tuple[int, int], int] = {}
    for (step, clust), rids in step_cluster_rids.items():
        n_pass = sum(1 for r in rids if rewards.get(r, 0) >= 0.5)
        step_cluster_sr[(step, clust)] = n_pass / len(rids) if rids else 0.0
        step_cluster_size[(step, clust)] = len(rids)

    minority_is_success = task_sr < 0.5

    weights: dict[int, float] = {}
    for rid, path in rollout_paths.items():
        early_clusters = path[:early_steps]
        if not early_clusters:
            weights[rid] = 1.0
            continue

        scores = []
        for t, clust in enumerate(early_clusters):
            sr = step_cluster_sr.get((t, clust), 0.5)
            size = step_cluster_size.get((t, clust), 1)

            outcome = sr if minority_is_success else (1.0 - sr)
            rarity = 1.0 / size

            if strategy == "minority_weighted":
                scores.append(outcome * rarity)
            elif strategy == "minority_only":
                scores.append(outcome)
            elif strategy == "branch_balanced":
                scores.append(rarity)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        weights[rid] = float(np.mean(scores)) if scores else 1.0

    n = len(weights)
    eps = 0.05 / max(n, 1)
    total = 0.0
    for rid in weights:
        weights[rid] = max(weights[rid], eps)
        total += weights[rid]
    for rid in weights:
        weights[rid] /= total

    return weights


def _build_step_cluster_table(
    indices: list[tuple[int, int]],
    labels: np.ndarray,
    rewards: dict[int, float],
    early_steps: int = 3,
) -> list[list[dict]]:
    """Build a per-step summary of clusters for diagnostic output.

    Returns list of length early_steps, each element is a list of dicts
    sorted by descending membership: {cluster, n, sr}.
    """
    from collections import defaultdict

    step_cluster_rids: dict[tuple[int, int], set[int]] = defaultdict(set)
    for (rid, step), lbl in zip(indices, labels):
        if step < early_steps:
            step_cluster_rids[(step, int(lbl))].add(rid)

    table: list[list[dict]] = []
    for t in range(early_steps):
        clusters_at_step = []
        for (step, clust), rids in step_cluster_rids.items():
            if step != t:
                continue
            n_pass = sum(1 for r in rids if rewards.get(r, 0) >= 0.5)
            clusters_at_step.append({
                "cluster": clust,
                "n": len(rids),
                "sr": n_pass / len(rids) if rids else 0.0,
            })
        clusters_at_step.sort(key=lambda x: -x["n"])
        table.append(clusters_at_step)
    return table


def diagnose_branch_weights(
    data_dir: str,
    out_dir: str,
    min_cluster_size: int = 5,
    noise_strategy: str = "outlier_bin",
    early_steps: int = 3,
) -> dict:
    """Compute and display biased-sampling diagnostics without running PRM eval.

    Returns the diagnostic dict (also saved to JSON) for programmatic use.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    base = Path(data_dir)
    action_dir = str(base / "embeddings" / "action")
    response_dir = str(base / "embeddings" / "response")

    task_meta = _discover_tasks(data_dir, action_dir)
    target_meta = [t for t in task_meta if _match_target_task(t["short_id"])]

    if not target_meta:
        print("ERROR: No target tasks found. Available tasks:")
        for t in task_meta:
            print(f"  {t['short_id']}")
        return {}

    print(f"Biased-subsample diagnostic: {len(target_meta)} target tasks")

    out = Path(out_dir) / "biased_subsample"
    out.mkdir(parents=True, exist_ok=True)

    diag: dict = {}
    rng_seed = 42

    for tmeta in target_meta:
        label = tmeta["short_id"]
        sl = _short_label(label)
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        action_npz = tmeta["npz_path"]
        response_npz = str(Path(response_dir) / Path(action_npz).name)
        if not Path(response_npz).exists():
            print(f"  WARNING: No response NPZ, skipping")
            continue

        act_mat, res_mat, indices = load_paired_embeddings(action_npz, response_npz)
        rewards = _load_rewards_from_files(tmeta["tdp_files"])
        all_rids = sorted(set(r for r, _ in indices) & set(rewards.keys()))
        n_total = len(all_rids)

        n_pass = sum(1 for r in all_rids if rewards.get(r, 0) >= 0.5)
        task_sr = n_pass / n_total if n_total else 0.0
        minority = "SUCCESS" if task_sr < 0.5 else "FAILURE"
        n_minority = n_pass if task_sr < 0.5 else (n_total - n_pass)

        S_full, _, _ = paired_similarity(act_mat, res_mat, "mean")
        ref_labels = hdbscan_cluster_affinity(
            S_full, min_cluster_size=min_cluster_size,
            noise_strategy=noise_strategy,
        )
        ref_K = int(ref_labels.max()) + 1
        _, rollout_paths = build_global_thought_graph(ref_labels, indices, rewards)
        rollout_paths = {r: rollout_paths[r] for r in all_rids if r in rollout_paths}

        print(f"  sr={task_sr:.2f}, K={ref_K}, n={n_total}, "
              f"minority={minority} ({n_minority}/{n_total})")

        cluster_table = _build_step_cluster_table(
            indices, ref_labels, rewards, early_steps,
        )
        print(f"\n  Early-step cluster map (steps 0..{early_steps - 1}):")
        for t, clusters in enumerate(cluster_table):
            parts = [f"c{c['cluster']}(n={c['n']}, sr={c['sr']:.2f})"
                     for c in clusters[:8]]
            suffix = f" +{len(clusters)-8} more" if len(clusters) > 8 else ""
            print(f"    Step {t}: {', '.join(parts)}{suffix}")

        task_diag: dict = {
            "label": label, "short_label": sl, "sr": task_sr,
            "minority": minority, "n_minority": n_minority,
            "n_total": n_total, "K": ref_K,
            "cluster_table": [[{
                "cluster": c["cluster"], "n": c["n"],
                "sr": round(c["sr"], 4),
            } for c in step_clusters] for step_clusters in cluster_table],
            "strategies": {},
        }

        rng = np.random.default_rng(rng_seed)

        for strat in BIASED_STRATEGIES:
            wts = compute_branch_weights(
                indices, ref_labels, rewards, rollout_paths,
                task_sr, strategy=strat, early_steps=early_steps,
            )

            rids_arr = np.array(sorted(wts.keys()))
            probs = np.array([wts[r] for r in rids_arr])

            draw_rng = np.random.default_rng(rng_seed)
            drawn = draw_rng.choice(
                rids_arr, size=min(BIASED_PRIMARY_N, len(rids_arr)),
                replace=False, p=probs,
            )
            drawn_set = set(drawn)

            n_minority_drawn = sum(
                1 for r in drawn
                if (rewards.get(r, 0) >= 0.5) == (task_sr < 0.5)
            )
            n_majority_drawn = len(drawn) - n_minority_drawn

            sorted_by_weight = sorted(wts.items(), key=lambda x: -x[1])
            top5 = sorted_by_weight[:5]
            bot5 = sorted_by_weight[-5:]

            w_vals = np.array(list(wts.values()))

            print(f"\n  Strategy: {strat}")
            print(f"    Top-5 by weight: "
                  + ", ".join(f"rid={r}(w={w:.4f})" for r, w in top5))
            print(f"    Bottom-5:        "
                  + ", ".join(f"rid={r}(w={w:.4f})" for r, w in bot5))
            print(f"    Weight dist: min={w_vals.min():.4f}, "
                  f"p25={np.percentile(w_vals, 25):.4f}, "
                  f"med={np.median(w_vals):.4f}, "
                  f"p75={np.percentile(w_vals, 75):.4f}, "
                  f"max={w_vals.max():.4f}")
            print(f"    N=80 draw: {n_minority_drawn}/{n_minority} "
                  f"minority ({minority}) rollouts captured "
                  f"({n_minority_drawn/max(n_minority,1):.0%}), "
                  f"{n_majority_drawn} majority")

            task_diag["strategies"][strat] = {
                "top5": [{"rid": int(r), "weight": round(w, 6)} for r, w in top5],
                "bottom5": [{"rid": int(r), "weight": round(w, 6)} for r, w in bot5],
                "weight_stats": {
                    "min": round(float(w_vals.min()), 6),
                    "p25": round(float(np.percentile(w_vals, 25)), 6),
                    "median": round(float(np.median(w_vals)), 6),
                    "p75": round(float(np.percentile(w_vals, 75)), 6),
                    "max": round(float(w_vals.max()), 6),
                },
                "draw_n80": {
                    "n_minority_drawn": n_minority_drawn,
                    "n_minority_total": n_minority,
                    "n_majority_drawn": n_majority_drawn,
                    "drawn_rids": sorted(int(r) for r in drawn),
                },
            }

        diag[label] = task_diag

    diag_path = out / "biased_subsample_diagnostic.json"
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Diagnostic saved to {diag_path}")
    print("Review the sampling plan above. If it looks good, rerun "
          "without --diagnose-only to execute PRM evaluation.")

    return diag


def run_biased_subsample_experiment(
    data_dir: str,
    out_dir: str,
    min_cluster_size: int = 5,
    noise_strategy: str = "outlier_bin",
    early_steps: int = 3,
) -> dict:
    """Run biased subsampling with nested sizes and PRM evaluation."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from tqdm import tqdm
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    base = Path(data_dir)
    action_dir = str(base / "embeddings" / "action")
    response_dir = str(base / "embeddings" / "response")

    task_meta = _discover_tasks(data_dir, action_dir)
    target_meta = [t for t in task_meta if _match_target_task(t["short_id"])]

    if not target_meta:
        print("ERROR: No target tasks found.")
        return {}

    n_configs = len(target_meta) * len(BIASED_STRATEGIES) * len(BIASED_NESTED_SIZES)
    print(f"Biased subsample experiment: {len(target_meta)} tasks × "
          f"{len(BIASED_STRATEGIES)} strategies × "
          f"{len(BIASED_NESTED_SIZES)} nested sizes = {n_configs} runs")

    out = Path(out_dir) / "biased_subsample"
    out.mkdir(parents=True, exist_ok=True)

    rng_seed = 42
    all_results: dict = {}
    pbar = tqdm(total=n_configs, desc="Biased subsample", unit="run")

    for tmeta in target_meta:
        label = tmeta["short_id"]
        action_npz = tmeta["npz_path"]
        response_npz = str(Path(response_dir) / Path(action_npz).name)
        if not Path(response_npz).exists():
            continue

        act_mat, res_mat, indices = load_paired_embeddings(action_npz, response_npz)
        rewards = _load_rewards_from_files(tmeta["tdp_files"])
        all_rids = sorted(set(r for r, _ in indices) & set(rewards.keys()))
        n_total = len(all_rids)

        n_pass = sum(1 for r in all_rids if rewards.get(r, 0) >= 0.5)
        task_sr = n_pass / n_total if n_total else 0.0

        S_full, _, _ = paired_similarity(act_mat, res_mat, "mean")
        ref_labels = hdbscan_cluster_affinity(
            S_full, min_cluster_size=min_cluster_size,
            noise_strategy=noise_strategy,
        )
        ref_K = int(ref_labels.max()) + 1
        n_steps = len(indices)

        _, rollout_paths = build_global_thought_graph(ref_labels, indices, rewards)
        rollout_paths = {r: rollout_paths[r] for r in all_rids if r in rollout_paths}

        print(f"\n  {label}: sr={task_sr:.2f}, K={ref_K}, n={n_total}")

        task_results: dict = {}

        for strat in BIASED_STRATEGIES:
            pbar.set_postfix(task=_short_label(label)[:15], strat=strat[:12])

            wts = compute_branch_weights(
                indices, ref_labels, rewards, rollout_paths,
                task_sr, strategy=strat, early_steps=early_steps,
            )

            rids_arr = np.array(sorted(wts.keys()))
            probs = np.array([wts[r] for r in rids_arr])

            draw_rng = np.random.default_rng(rng_seed)
            n_draw = min(BIASED_PRIMARY_N, len(rids_arr))
            primary_draw = draw_rng.choice(
                rids_arr, size=n_draw, replace=False, p=probs,
            )

            weight_order = sorted(
                range(len(primary_draw)),
                key=lambda i: -wts[primary_draw[i]],
            )
            ordered_rids = [int(primary_draw[j]) for j in weight_order]

            for n_sub in BIASED_NESTED_SIZES:
                if n_sub > len(ordered_rids):
                    pbar.update(1)
                    continue

                sub_rids = sorted(ordered_rids[:n_sub])
                sub_rid_set = set(sub_rids)

                sub_mask = np.array([
                    i for i, (r, _) in enumerate(indices) if r in sub_rid_set
                ])
                ho_mask = np.array([
                    i for i, (r, _) in enumerate(indices) if r not in sub_rid_set
                ])

                sub_act, sub_res = act_mat[sub_mask], res_mat[sub_mask]
                sub_indices = [indices[i] for i in sub_mask]

                S_sub, _, _ = paired_similarity(sub_act, sub_res, "mean")
                effective_mcs = max(2, min(min_cluster_size, n_sub // 5))
                sub_labels = hdbscan_cluster_affinity(
                    S_sub, min_cluster_size=effective_mcs,
                    noise_strategy=noise_strategy,
                )
                K = int(sub_labels.max()) + 1

                if len(ho_mask) > 0:
                    ho_act, ho_res = act_mat[ho_mask], res_mat[ho_mask]
                    ho_labels = _assign_heldout_to_clusters(
                        sub_act, sub_res, sub_labels, ho_act, ho_res,
                    )
                    merged_labels = np.concatenate([sub_labels, ho_labels])
                    merged_order = np.concatenate([sub_mask, ho_mask])
                    merged_indices = sub_indices + [indices[i] for i in ho_mask]
                else:
                    merged_labels = sub_labels
                    merged_order = sub_mask
                    merged_indices = sub_indices

                full_labels = np.empty(n_steps, dtype=merged_labels.dtype)
                full_labels[merged_order] = merged_labels

                ari = float(adjusted_rand_score(ref_labels, full_labels))
                nmi = float(normalized_mutual_info_score(ref_labels, full_labels))

                _, rp = build_global_thought_graph(
                    merged_labels, merged_indices, rewards,
                )
                eval_rids = sorted(set(rp.keys()) & set(rewards.keys()))
                rp = {r: rp[r] for r in eval_rids}
                outcome = np.array([
                    1.0 if rewards.get(r, 0) >= 0.5 else 0.0 for r in eval_rids
                ])

                td = TaskData(
                    label=label, domain=tmeta["domain"], task_id=tmeta["task_id"],
                    labels=merged_labels, indices=merged_indices,
                    rewards={r: rewards[r] for r in eval_rids},
                    rollout_paths=rp, K=K,
                    all_rids=eval_rids, outcome=outcome,
                    step_embeddings=None, rid_to_emb_rows=None,
                )

                level_results: dict = {}
                skip_levels = {"level3_fasttext"}
                for level_name, (prm_class, prm_kwargs) in LEVELS.items():
                    if level_name in skip_levels:
                        continue
                    cv = run_cv_for_task(td, prm_class, prm_kwargs)
                    level_results[level_name] = cv

                key = f"{strat}_mcs{min_cluster_size}_n{n_sub}"
                if key not in task_results:
                    task_results[key] = []
                task_results[key].append({
                    "K": K, "ref_K": ref_K, "ari": ari, "nmi": nmi,
                    "strategy": strat, "n_sub": n_sub,
                    "levels": level_results,
                })
                pbar.update(1)

        all_results[label] = task_results

    pbar.close()

    results_path = out / "biased_subsample_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved biased subsample results to {results_path}")

    plot_biased_vs_random(all_results, out, data_dir)
    return all_results


def plot_biased_vs_random(
    biased_results: dict,
    out_dir: str | Path,
    data_dir: str | None = None,
):
    """Plot biased strategies vs random baseline at nested sample sizes."""
    import re

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    random_results: dict | None = None
    if data_dir:
        for subdir in ["prm_noise_bin", "prm"]:
            rpath = Path(data_dir) / "analysis" / subdir / "subsample" / "subsample_results.json"
            if rpath.exists():
                with open(rpath) as f:
                    random_results = json.load(f)
                break

    strat_colors = {
        "minority_weighted": "#d62728",
        "minority_only": "#ff7f0e",
        "branch_balanced": "#2ca02c",
        "random": "#1f77b4",
    }
    strat_markers = {
        "minority_weighted": "D",
        "minority_only": "s",
        "branch_balanced": "^",
        "random": "o",
    }

    for task_label, task_data in biased_results.items():
        sl = _short_label(task_label)

        strat_size_metrics: dict[str, dict[int, dict]] = {}
        for key, repeats in task_data.items():
            m = re.match(r"(.+)_mcs\d+_n(\d+)$", key)
            if not m:
                continue
            strat_name = m.group(1)
            n_sub = int(m.group(2))

            if strat_name not in strat_size_metrics:
                strat_size_metrics[strat_name] = {}

            for rep in repeats:
                if n_sub not in strat_size_metrics[strat_name]:
                    strat_size_metrics[strat_name][n_sub] = {
                        "K": [], "ari": [], "nmi": [], "levels": {},
                    }
                entry = strat_size_metrics[strat_name][n_sub]
                entry["K"].append(rep["K"])
                entry["ari"].append(rep["ari"])
                entry["nmi"].append(rep["nmi"])

                for lname, ldata in rep.get("levels", {}).items():
                    if lname not in entry["levels"]:
                        entry["levels"][lname] = {"auroc": [], "bon8": []}
                    strats = ldata.get("strategies", {})
                    final = strats.get("final", {})
                    if "auroc" in final:
                        entry["levels"][lname]["auroc"].append(final["auroc"])
                    bon = final.get("bon", {})
                    bon8 = bon.get(8) or bon.get("8")
                    if bon8 is not None:
                        if isinstance(bon8, dict):
                            bon8 = bon8.get("acc", bon8.get("accuracy"))
                        if bon8 is not None:
                            entry["levels"][lname]["bon8"].append(bon8)

        if random_results and task_label in random_results:
            rtask = random_results[task_label]
            strat_size_metrics["random"] = {}
            for rkey, rrepeats in rtask.items():
                rm = re.match(r"mcs\d+_n(\d+)$", rkey)
                if not rm:
                    continue
                rn = int(rm.group(1))
                if rn not in BIASED_NESTED_SIZES:
                    continue

                strat_size_metrics["random"][rn] = {
                    "K": [], "ari": [], "nmi": [], "levels": {},
                }
                entry = strat_size_metrics["random"][rn]
                for rrep in rrepeats:
                    entry["K"].append(rrep.get("K", 0))
                    for lname, ldata in rrep.get("levels", {}).items():
                        if lname not in entry["levels"]:
                            entry["levels"][lname] = {"auroc": [], "bon8": []}
                        strats = ldata.get("strategies", {})
                        final = strats.get("final", {})
                        if "auroc" in final:
                            entry["levels"][lname]["auroc"].append(final["auroc"])
                        bon = final.get("bon", {})
                        bon8 = bon.get(8) or bon.get("8")
                        if bon8 is not None:
                            if isinstance(bon8, dict):
                                bon8 = bon8.get("acc", bon8.get("accuracy"))
                            if bon8 is not None:
                                entry["levels"][lname]["bon8"].append(bon8)

        all_levels = set()
        for sd in strat_size_metrics.values():
            for nd in sd.values():
                all_levels.update(nd["levels"].keys())
        all_levels = sorted(all_levels)
        if not all_levels:
            all_levels = ["level0_markov", "level1_likelihood", "level2_logreg"]
        n_levels = len(all_levels)

        fig, axes = plt.subplots(4, max(n_levels, 1),
                                 figsize=(5 * max(n_levels, 1), 16),
                                 squeeze=False)

        for li, lname in enumerate(all_levels):
            for row, (metric, ylabel) in enumerate([
                ("auroc", "AUROC"), ("bon8", "BoN@8"),
            ]):
                ax = axes[row][li]
                for sname in ["random"] + BIASED_STRATEGIES:
                    if sname not in strat_size_metrics:
                        continue
                    sizes_dict = strat_size_metrics[sname]
                    sizes = sorted(sizes_dict.keys())
                    means, stds = [], []
                    valid_sizes = []
                    for ns in sizes:
                        vals = sizes_dict[ns]["levels"].get(lname, {}).get(metric, [])
                        if vals:
                            means.append(np.mean(vals))
                            stds.append(np.std(vals) / max(np.sqrt(len(vals)), 1))
                            valid_sizes.append(ns)
                    if not valid_sizes:
                        continue
                    ax.errorbar(
                        valid_sizes, means, yerr=stds,
                        label=sname, color=strat_colors.get(sname, "gray"),
                        marker=strat_markers.get(sname, "."),
                        markersize=6, linewidth=1.5, capsize=3,
                    )
                ax.set_xlabel("N (rollouts in subsample)")
                ax.set_ylabel(ylabel)
                lname_short = lname.replace("level", "L").replace("_", " ")
                ax.set_title(f"{lname_short}")
                ax.grid(True, alpha=0.2)
                if row == 0 and li == 0:
                    ax.legend(fontsize=7, loc="lower right")

        for li in range(max(n_levels, 1)):
            ax_ari = axes[2][li]
            ax_k = axes[3][li]
            if li >= len(all_levels):
                ax_ari.set_visible(False)
                ax_k.set_visible(False)
                continue

            for sname in ["random"] + BIASED_STRATEGIES:
                if sname not in strat_size_metrics:
                    continue
                sizes_dict = strat_size_metrics[sname]
                sizes = sorted(sizes_dict.keys())

                ari_means, nmi_means = [], []
                k_means = []
                valid_sizes_ari, valid_sizes_k = [], []

                for ns in sizes:
                    ari_vals = sizes_dict[ns].get("ari", [])
                    nmi_vals = sizes_dict[ns].get("nmi", [])
                    k_vals = sizes_dict[ns].get("K", [])
                    if ari_vals:
                        ari_means.append(np.mean(ari_vals))
                        nmi_means.append(np.mean(nmi_vals) if nmi_vals else 0)
                        valid_sizes_ari.append(ns)
                    if k_vals:
                        k_means.append(np.mean(k_vals))
                        valid_sizes_k.append(ns)

                color = strat_colors.get(sname, "gray")
                marker = strat_markers.get(sname, ".")

                if valid_sizes_ari and li == 0:
                    ax_ari.plot(
                        valid_sizes_ari, ari_means,
                        label=f"{sname} ARI", color=color,
                        marker=marker, markersize=5, linewidth=1.5,
                    )
                    ax_ari.plot(
                        valid_sizes_ari, nmi_means,
                        label=f"{sname} NMI", color=color,
                        marker=marker, markersize=5, linewidth=1.5,
                        linestyle="--",
                    )
                if valid_sizes_k and li == 0:
                    ax_k.plot(
                        valid_sizes_k, k_means,
                        label=sname, color=color,
                        marker=marker, markersize=5, linewidth=1.5,
                    )

            if li == 0:
                ax_ari.set_xlabel("N")
                ax_ari.set_ylabel("ARI / NMI")
                ax_ari.set_title("Cluster convergence vs reference")
                ax_ari.grid(True, alpha=0.2)
                ax_ari.legend(fontsize=6, ncol=2)

                ax_k.set_xlabel("N")
                ax_k.set_ylabel("K (clusters)")
                ax_k.set_title("Cluster count")
                ax_k.grid(True, alpha=0.2)
                ax_k.legend(fontsize=7)
            else:
                ax_ari.set_visible(False)
                ax_k.set_visible(False)

        fig.suptitle(
            f"Biased vs Random Subsampling — {sl}",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        safe_name = sl.replace(" ", "_").replace("/", "_")
        fig.savefig(out / f"biased_vs_random_{safe_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out / f'biased_vs_random_{safe_name}.png'}")


# ── Fork-and-continue minority branch exploration ────────────────────

FORK_STRATEGIES = ["minority_weighted", "minority_early", "early_rare"]
FORK_BUDGET_TOTAL = 1000
FORK_MIN_CONTINUATIONS = 2


FORK_MAX_PROGRESS = 0.75


def select_fork_points(
    indices: list[tuple[int, int]],
    labels: np.ndarray,
    rewards: dict[int, float],
    rollout_paths: dict[int, list[int]],
    task_sr: float,
    strategy: str = "minority_weighted",
    budget: int = 111,
    max_progress: float = FORK_MAX_PROGRESS,
) -> list[dict]:
    """Select (rollout_id, step) fork points and allocate continuations.

    For minority_weighted and minority_early, uses transition-based scoring:
    only considers minority-outcome rollouts, scores the step *before* the
    minority signal increases (the decision point), so forks happen where the
    trajectory could still have taken a different path.

    Returns a list of dicts: {rid, step, cluster, cluster_size, cluster_sr,
    score, n_continuations}.
    """
    from collections import defaultdict

    step_cluster_rids: dict[tuple[int, int], set[int]] = defaultdict(set)
    for (rid, step), lbl in zip(indices, labels):
        step_cluster_rids[(step, int(lbl))].add(rid)

    step_cluster_sr: dict[tuple[int, int], float] = {}
    step_cluster_size: dict[tuple[int, int], int] = {}
    for (step, clust), rids in step_cluster_rids.items():
        n_pass = sum(1 for r in rids if rewards.get(r, 0) >= 0.5)
        step_cluster_sr[(step, clust)] = n_pass / len(rids) if rids else 0.0
        step_cluster_size[(step, clust)] = len(rids)

    minority_is_success = task_sr < 0.5

    avg_len = np.mean([len(p) for p in rollout_paths.values()])
    global_max_step = int(avg_len * max_progress)

    def _minority_raw(sr: float) -> float:
        return sr if minority_is_success else (1.0 - sr)

    candidates: list[dict] = []

    if strategy in ("minority_weighted", "minority_early"):
        # Transition-based: fork at the step before minority signal jumps.
        # Only consider rollouts that ended in the minority outcome.
        minority_rids = {
            r for r in rollout_paths
            if (rewards.get(r, 0) >= 0.5) == minority_is_success
        }
        for rid in minority_rids:
            path = rollout_paths[rid]
            if len(path) < 2:
                continue
            ms_prev = _minority_raw(
                step_cluster_sr.get((0, path[0]), 0.5)
            )
            for t in range(len(path) - 1):
                if t > global_max_step:
                    break
                clust_now = path[t]
                clust_next = path[t + 1]
                ms_now = ms_prev
                ms_next = _minority_raw(
                    step_cluster_sr.get((t + 1, clust_next), 0.5)
                )
                delta = ms_next - ms_now
                ms_prev = ms_next

                if delta <= 0:
                    continue

                size = step_cluster_size.get((t, clust_now), 1)
                sr = step_cluster_sr.get((t, clust_now), 0.5)
                step_decay = 1.0 / (t + 1)
                rarity = 1.0 / size

                if strategy == "minority_weighted":
                    score = delta * rarity * step_decay
                else:
                    score = delta * step_decay

                candidates.append({
                    "rid": rid, "step": t, "cluster": clust_now,
                    "cluster_size": size, "cluster_sr": sr,
                    "score": score,
                })

    elif strategy == "early_rare":
        for rid, path in rollout_paths.items():
            for t, clust in enumerate(path):
                if t > global_max_step:
                    break
                size = step_cluster_size.get((t, clust), 1)
                sr = step_cluster_sr.get((t, clust), 0.5)
                rarity = 1.0 / size
                step_decay = 1.0 / (t + 1)
                score = rarity * step_decay

                if score <= 0:
                    continue
                candidates.append({
                    "rid": rid, "step": t, "cluster": clust,
                    "cluster_size": size, "cluster_sr": sr,
                    "score": score,
                })
    else:
        raise ValueError(f"Unknown fork strategy: {strategy}")

    if not candidates:
        return []

    unique_points: dict[tuple[int, int], dict] = {}
    for c in candidates:
        key = (c["rid"], c["step"])
        if key not in unique_points or c["score"] > unique_points[key]["score"]:
            unique_points[key] = c

    scored = sorted(unique_points.values(), key=lambda x: -x["score"])

    total_score = sum(c["score"] for c in scored)
    if total_score == 0:
        return []

    for c in scored:
        c["n_continuations"] = max(
            FORK_MIN_CONTINUATIONS,
            int(round(budget * c["score"] / total_score)),
        )

    allocated = sum(c["n_continuations"] for c in scored)
    result = []
    running = 0
    for c in scored:
        if running + c["n_continuations"] > budget:
            remaining = budget - running
            if remaining >= FORK_MIN_CONTINUATIONS:
                c["n_continuations"] = remaining
                result.append(c)
                running += remaining
            break
        result.append(c)
        running += c["n_continuations"]

    return result


def diagnose_fork_points(
    data_dir: str,
    out_dir: str,
    min_cluster_size: int = 5,
    noise_strategy: str = "outlier_bin",
) -> dict:
    """Compute and display fork-point diagnostics without running any forks."""
    base = Path(data_dir)
    action_dir = str(base / "embeddings" / "action")
    response_dir = str(base / "embeddings" / "response")

    task_meta = _discover_tasks(data_dir, action_dir)
    target_meta = [t for t in task_meta if _match_target_task(t["short_id"])]

    if not target_meta:
        print("ERROR: No target tasks found.")
        return {}

    n_tasks = len(target_meta)
    n_strats = len(FORK_STRATEGIES)
    per_task_budget = FORK_BUDGET_TOTAL // n_tasks
    per_strat_budget = per_task_budget // n_strats

    print(f"Fork diagnostic: {n_tasks} tasks, {n_strats} strategies, "
          f"budget={FORK_BUDGET_TOTAL} "
          f"({per_strat_budget} per strategy-task pair)")

    out = Path(out_dir) / "minority_fork"
    out.mkdir(parents=True, exist_ok=True)

    diag: dict = {}

    for tmeta in target_meta:
        label = tmeta["short_id"]
        sl = _short_label(label)

        action_npz = tmeta["npz_path"]
        response_npz = str(Path(response_dir) / Path(action_npz).name)
        if not Path(response_npz).exists():
            print(f"  WARNING: No response NPZ for {label}, skipping")
            continue

        act_mat, res_mat, indices = load_paired_embeddings(action_npz, response_npz)
        rewards = _load_rewards_from_files(tmeta["tdp_files"])
        all_rids = sorted(set(r for r, _ in indices) & set(rewards.keys()))
        n_total = len(all_rids)
        n_pass = sum(1 for r in all_rids if rewards.get(r, 0) >= 0.5)
        task_sr = n_pass / n_total if n_total else 0.0
        minority = "SUCCESS" if task_sr < 0.5 else "FAILURE"
        n_minority = n_pass if task_sr < 0.5 else (n_total - n_pass)

        S_full, _, _ = paired_similarity(act_mat, res_mat, "mean")
        ref_labels = hdbscan_cluster_affinity(
            S_full, min_cluster_size=min_cluster_size,
            noise_strategy=noise_strategy,
        )
        ref_K = int(ref_labels.max()) + 1
        _, rollout_paths = build_global_thought_graph(ref_labels, indices, rewards)
        rollout_paths = {r: rollout_paths[r] for r in all_rids if r in rollout_paths}

        print(f"\n{'='*60}")
        print(f"  {sl} — {label}")
        avg_len = np.mean([len(p) for p in rollout_paths.values()])
        max_fork_step = int(avg_len * FORK_MAX_PROGRESS)
        print(f"  sr={task_sr:.2f}, K={ref_K}, n={n_total}, "
              f"minority={minority} ({n_minority}/{n_total})")
        print(f"  avg rollout length={avg_len:.1f}, "
              f"max fork step={max_fork_step} (75%)")
        print(f"{'='*60}")

        cluster_table = _build_step_cluster_table(
            indices, ref_labels, rewards, early_steps=5,
        )
        print(f"\n  Cluster map (steps 0..4):")
        for t, clusters in enumerate(cluster_table):
            parts = [f"c{c['cluster']}(n={c['n']}, sr={c['sr']:.2f})"
                     for c in clusters[:6]]
            suffix = f" +{len(clusters)-6} more" if len(clusters) > 6 else ""
            print(f"    Step {t}: {', '.join(parts)}{suffix}")

        task_diag: dict = {
            "label": label, "short_label": sl, "sr": task_sr,
            "minority": minority, "n_minority": n_minority,
            "n_total": n_total, "K": ref_K,
            "task_id": tmeta["task_id"], "domain": tmeta["domain"],
            "strategies": {},
        }

        for strat in FORK_STRATEGIES:
            forks = select_fork_points(
                indices, ref_labels, rewards, rollout_paths,
                task_sr, strategy=strat, budget=per_strat_budget,
            )

            total_cont = sum(f["n_continuations"] for f in forks)
            unique_rids = set(f["rid"] for f in forks)
            minority_rids_used = sum(
                1 for r in unique_rids
                if (rewards.get(r, 0) >= 0.5) == (task_sr < 0.5)
            )

            from collections import defaultdict as _ddict
            _sc = _ddict(lambda: {"n_conts": 0, "rids": set(),
                                  "size": 0, "sr": 0.0})
            for fp in forks:
                k = (fp["step"], fp["cluster"])
                _sc[k]["n_conts"] += fp["n_continuations"]
                _sc[k]["rids"].add(fp["rid"])
                _sc[k]["size"] = fp["cluster_size"]
                _sc[k]["sr"] = fp["cluster_sr"]
            _sorted = sorted(_sc.items(), key=lambda x: (x[0][0], x[0][1]))

            hdr = "  Step \u2502 Cluster \u2502 Size \u2502   SR  \u2502 Rids forked \u2502 Conts"
            sep = f"  {'\u2500'*58}"
            print(f"\n  Strategy: {strat} "
                  f"({total_cont} conts, {len(forks)} fork points)")
            print(sep)
            print(hdr)
            print(sep)
            for (step, clust), info in _sorted:
                rids_s = ",".join(str(r) for r in sorted(info["rids"]))
                if len(rids_s) > 25:
                    rids_s = rids_s[:22] + "..."
                print(f"  {step:4d} \u2502 c{clust:<5d} \u2502 {info['size']:4d} "
                      f"\u2502 {info['sr']:5.2f} \u2502 {rids_s:<11s} "
                      f"\u2502 {info['n_conts']}")
            print(sep)
            steps_used = sorted(set(f["step"] for f in forks))
            print(f"  Unique rids: {len(unique_rids)}, "
                  f"minority sources: {minority_rids_used}/{len(unique_rids)}, "
                  f"steps: {steps_used}")
            print(f"    Steps used: {steps_used}")

            task_diag["strategies"][strat] = {
                "fork_points": [{
                    "rid": int(f["rid"]), "step": f["step"],
                    "cluster": int(f["cluster"]),
                    "cluster_size": f["cluster_size"],
                    "cluster_sr": round(f["cluster_sr"], 4),
                    "score": round(f["score"], 6),
                    "n_continuations": f["n_continuations"],
                } for f in forks],
                "total_continuations": total_cont,
                "n_fork_points": len(forks),
                "steps_used": steps_used,
            }

        diag[label] = task_diag

    diag_path = out / "fork_diagnostic.json"
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)

    total_all = sum(
        strat_data["total_continuations"]
        for td in diag.values()
        for strat_data in td["strategies"].values()
    )
    print(f"\n{'='*60}")
    print(f"Total continuations across all tasks/strategies: {total_all}")
    print(f"Diagnostic saved to {diag_path}")
    print("Review the fork plan above. Rerun without --diagnose-only "
          "to execute the forks.")

    return diag


def _suppress_noisy_loggers():
    """Silence orchestrator / LLM debug spam."""
    import logging as _logging
    import sys as _sys
    for noisy in (
        "tau2", "LiteLLM", "litellm", "openai", "httpx", "httpcore",
        "urllib3", "google", "grpc", "absl",
        "src.tau2",
    ):
        _logging.getLogger(noisy).setLevel(_logging.ERROR)
    try:
        import litellm
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
    except ImportError:
        pass
    try:
        from loguru import logger as _loguru_logger
        _loguru_logger.remove()
        _loguru_logger.add(_sys.stderr, level="ERROR")
    except ImportError:
        pass


def run_fork_experiment(
    data_dir: str,
    out_dir: str,
    min_cluster_size: int = 5,
    noise_strategy: str = "outlier_bin",
    max_concurrency: int = 10,
) -> dict:
    """Execute fork-and-continue using the fork diagnostic plan."""
    import threading
    import warnings
    from concurrent.futures import ThreadPoolExecutor, as_completed

    warnings.filterwarnings("ignore", category=FutureWarning)

    diag = diagnose_fork_points(
        data_dir, out_dir,
        min_cluster_size=min_cluster_size,
        noise_strategy=noise_strategy,
    )
    if not diag:
        return {}

    _suppress_noisy_loggers()

    from cluster_predictive_power import (
        cold_fork_from_tdp,
        evaluate_branch,
        _find_tdp_path,
    )
    from tau2.run import get_tasks
    from src.tau2.variance_guided_runner import _serialize_messages
    from tqdm import tqdm

    config_path = Path(data_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            "model": "qwen3-next-80b-thinking",
            "user_model": "gpt-4o-mini",
            "temperature": 0.8,
            "max_steps": 100,
            "max_errors": 10,
        }

    fork_out = Path(out_dir) / "minority_fork"
    fork_out.mkdir(parents=True, exist_ok=True)

    # -- load existing results to skip completed forks on retry --
    results_jsonl = fork_out / "fork_results.jsonl"
    completed_ids: set[tuple[str, str]] = set()
    prior_results: list[dict] = []
    if results_jsonl.exists():
        for line in open(results_jsonl):
            r = json.loads(line)
            if not r.get("error"):
                key = (r.get("strategy", ""), r.get("fork_id", ""))
                completed_ids.add(key)
                prior_results.append(r)

    # -- flatten all fork jobs into a list of work dicts --
    all_jobs: list[dict] = []
    fork_counter = 0
    task_objs: dict = {}

    for task_label, task_diag in diag.items():
        domain = task_diag["domain"]
        task_id = task_diag["task_id"]

        bare_id = task_id.split("/", 1)[1] if "/" in task_id else task_id
        matched = get_tasks(
            task_set_name=domain,
            task_split_name="base",
            task_ids=[bare_id],
        )
        if not matched:
            print(f"WARNING: Could not find task {task_id}, skipping")
            continue
        task_objs[task_label] = matched[0]

        for strat, strat_data in task_diag["strategies"].items():
            strat_out = fork_out / strat / domain
            strat_out.mkdir(parents=True, exist_ok=True)

            for fp in strat_data["fork_points"]:
                rid = fp["rid"]
                step = fp["step"]
                n_cont = fp["n_continuations"]

                tdp_path = _find_tdp_path(data_dir, task_id, rid)
                if tdp_path is None:
                    print(f"WARNING: TDP not found for {task_id} "
                          f"rid={rid}, skipping")
                    continue

                for ci in range(n_cont):
                    fork_id = f"{rid}_s{step}_f{ci}"
                    all_jobs.append({
                        "task_label": task_label,
                        "domain": domain,
                        "task_id": task_id,
                        "strat": strat,
                        "rid": rid,
                        "step": step,
                        "ci": ci,
                        "fork_id": fork_id,
                        "seed": 7000 + fork_counter,
                        "tdp_path": tdp_path,
                        "strat_out": strat_out,
                    })
                    fork_counter += 1

    jobs = [j for j in all_jobs
            if (j["strat"], j["fork_id"]) not in completed_ids]

    if completed_ids:
        print(f"\nSkipping {len(completed_ids)} already-completed forks "
              f"(retrying {len(jobs)} errors/missing)")
    print(f"Running {len(jobs)} forks with concurrency={max_concurrency}")

    # -- worker function --
    write_lock = threading.Lock()

    def _run_one_fork(job: dict) -> dict:
        task_obj = task_objs[job["task_label"]]
        orch = cold_fork_from_tdp(
            tdp_path=job["tdp_path"],
            step=job["step"],
            new_seed=job["seed"],
            task=task_obj,
            domain=job["domain"],
            model=config.get("model", "qwen3-next-80b-thinking"),
            user_model=config.get("user_model", "gpt-4o-mini"),
            n_samples=1,
            temperature=config.get("temperature", 0.8),
            max_steps=config.get("max_steps", 100),
            max_errors=config.get("max_errors", 10),
        )

        while not orch.done:
            orch.step()

        reward, reward_info = evaluate_branch(orch, task_obj, job["domain"])

        result = {
            "fork_id": job["fork_id"],
            "parent_rid": job["rid"],
            "fork_step": job["step"],
            "continuation_idx": job["ci"],
            "seed": job["seed"],
            "strategy": job["strat"],
            "task_label": job["task_label"],
            "reward": reward,
            "reward_info": reward_info,
            "termination_reason": (
                orch.termination_reason.value
                if orch.termination_reason else "unknown"
            ),
            "total_steps": orch.step_count,
        }

        tdp_save_path = job["strat_out"] / f"{job['fork_id']}_tdp.json"
        messages = orch.get_trajectory()
        tdp_data = {
            "task_id": job["task_id"],
            "domain": job["domain"],
            "fork_id": job["fork_id"],
            "parent_rid": job["rid"],
            "fork_step": job["step"],
            "strategy": job["strat"],
            "seed": job["seed"],
            "reward": reward,
            "messages": _serialize_messages(messages),
        }
        with write_lock:
            with open(tdp_save_path, "w") as f:
                json.dump(tdp_data, f, indent=2)

        return result

    # -- run with ThreadPoolExecutor --
    n_pass = 0
    n_fail = 0
    n_err = 0
    all_results_flat: list[dict] = []

    results_jsonl = fork_out / "fork_results.jsonl"
    results_file = open(results_jsonl, "a")

    pbar = tqdm(total=len(jobs), desc="Forks", unit="fork")

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(_run_one_fork, job): job for job in jobs
        }
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                result = fut.result()
                all_results_flat.append(result)
                with write_lock:
                    results_file.write(json.dumps(result) + "\n")
                    results_file.flush()
                if (result.get("reward") or 0) >= 0.5:
                    n_pass += 1
                else:
                    n_fail += 1
            except Exception as e:
                n_err += 1
                tqdm.write(f"ERROR {job['fork_id']}: {e}")
                err_result = {
                    "fork_id": job["fork_id"],
                    "parent_rid": job["rid"],
                    "fork_step": job["step"],
                    "continuation_idx": job["ci"],
                    "seed": job["seed"],
                    "strategy": job["strat"],
                    "task_label": job["task_label"],
                    "reward": None,
                    "error": str(e),
                }
                all_results_flat.append(err_result)
                with write_lock:
                    results_file.write(json.dumps(err_result) + "\n")
                    results_file.flush()

            pbar.update(1)
            pbar.set_postfix(P=n_pass, F=n_fail, E=n_err)

    pbar.close()
    results_file.close()

    # -- merge prior successful results with new results --
    merged = prior_results + all_results_flat

    all_results: dict = {}
    for r in merged:
        tl = r.get("task_label", "unknown")
        st = r.get("strategy", "unknown")
        all_results.setdefault(tl, {}).setdefault(st, []).append(r)

    total_pass = sum(1 for r in merged
                     if not r.get("error") and (r.get("reward") or 0) >= 0.5)
    total_fail = sum(1 for r in merged
                     if not r.get("error") and (r.get("reward") or 0) < 0.5)
    total_err = sum(1 for r in merged if r.get("error"))

    print(f"\n{'='*60}")
    print(f"Combined results (prior + this run):")
    for tl, strats in sorted(all_results.items()):
        sl = _short_label(tl)
        for st, results in sorted(strats.items()):
            sp = sum(1 for r in results
                     if not r.get("error") and (r.get("reward") or 0) >= 0.5)
            ok = sum(1 for r in results if not r.get("error"))
            errs = sum(1 for r in results if r.get("error"))
            err_s = f" +{errs}err" if errs else ""
            print(f"  {sl}/{st}: {sp}/{ok} passed "
                  f"({sp/max(ok,1):.0%}){err_s}")

    results_path = fork_out / "fork_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved fork results to {results_path}")
    print(f"Total: {total_pass} pass, {total_fail} fail, "
          f"{total_err} errors ({len(merged)} total)")

    return all_results


# =====================================================================
#  Fork re-cluster + flow diagrams
# =====================================================================


def _extract_fork_steps(messages: list[dict]) -> list[tuple[str, str]]:
    """Extract (action_text, response_text) pairs from a forked TDP messages list.

    Each TDP "step" is an assistant turn. The action is the assistant's
    content + tool_calls; the response is the next non-assistant message.
    """
    from src.prm.embeddings import _serialize_tool_calls

    steps: list[tuple[str, str]] = []
    n = len(messages)

    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        content = (m.get("content") or "").strip()
        tool_calls = m.get("tool_calls")
        parts: list[str] = []
        if content:
            parts.append(content)
        if tool_calls:
            tcs = []
            for tc in tool_calls:
                fn = tc.get("function", tc)
                tcs.append({
                    "name": fn.get("name", "unknown"),
                    "arguments": fn.get("arguments", {}),
                })
            parts.append(_serialize_tool_calls(tcs))
        action_text = "\n".join(parts) if parts else "<empty>"

        response_text = "<empty>"
        for j in range(i + 1, n):
            nxt = messages[j]
            if nxt.get("role") in ("tool", "user"):
                rc = nxt.get("content", "")
                if isinstance(rc, (dict, list)):
                    rc = json.dumps(rc, sort_keys=True)
                response_text = str(rc).strip() or "<empty>"
                break

        steps.append((action_text, response_text))

    return steps


def _embed_texts_sync(
    texts: list[str], model: str = "text-embedding-3-large",
    max_workers: int = 4, chunk_size: int = 128,
) -> np.ndarray:
    """Embed a list of texts using the synchronous OpenAI API. Returns [N x dim]."""
    import os
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    except ImportError:
        pass

    client = OpenAI()
    dim = 3072
    embeddings = np.zeros((len(texts), dim), dtype=np.float32)

    chunks = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        indices = list(range(i, i + len(chunk)))
        chunks.append((indices, chunk))

    def _embed_chunk(indices, txts):
        import time as _time
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=model, input=txts)
                return [(indices[item.index], np.array(item.embedding, dtype=np.float32))
                        for item in resp.data]
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e):
                    _time.sleep(2 ** attempt)
                    continue
                raise
        resp = client.embeddings.create(model=model, input=txts)
        return [(indices[item.index], np.array(item.embedding, dtype=np.float32))
                for item in resp.data]

    from tqdm import tqdm
    pbar = tqdm(total=len(texts), desc="Embedding", unit="text")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_embed_chunk, idxs, txts): ci
            for ci, (idxs, txts) in enumerate(chunks)
        }
        for fut in as_completed(futures):
            results = fut.result()
            for idx, vec in results:
                embeddings[idx] = vec
            pbar.update(len(results))

    pbar.close()
    return embeddings


def recluster_fork_strategies(
    data_dir: str,
    out_dir: str,
    min_cluster_size: int = 5,
    noise_strategy: str = "outlier_bin",
    tool_arg_beta: float = 0.0,
):
    """Re-cluster original + forked rollouts per strategy, render flow diagrams."""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    from src.prm.thought_dag import render_trajectory_flow

    fork_out = Path(out_dir) / "minority_fork"
    results_jsonl = fork_out / "fork_results.jsonl"
    if not results_jsonl.exists():
        print("No fork results found. Run --fork first.")
        return

    completed: dict[str, list[dict]] = {}
    for line in open(results_jsonl):
        r = json.loads(line)
        if r.get("error"):
            continue
        strat = r.get("strategy", "unknown")
        completed.setdefault(strat, []).append(r)

    print(f"Fork results: {sum(len(v) for v in completed.values())} successful "
          f"across {len(completed)} strategies")

    base = Path(data_dir)
    action_dir = str(base / "embeddings" / "action")
    response_dir = str(base / "embeddings" / "response")

    task_meta = _discover_tasks(data_dir, action_dir)
    if not task_meta:
        print("No tasks discovered")
        return

    # Group fork results by (task_label, strategy) for fast lookup
    fork_by_task_strat: dict[str, dict[str, list[dict]]] = {}
    for strat, strat_results in completed.items():
        for r in strat_results:
            tl = r.get("task_label", "")
            fork_by_task_strat.setdefault(tl, {}).setdefault(strat, []).append(r)

    # Identify which tasks have fork data
    fork_task_labels = set(fork_by_task_strat.keys())
    active_meta = [t for t in task_meta if t["short_id"] in fork_task_labels]
    if not active_meta:
        print(f"No task metadata matches fork task labels: {fork_task_labels}")
        return

    print(f"Tasks with fork data: {[t['short_id'] for t in active_meta]}")

    def _safe_name(label: str) -> str:
        return label.replace("/", "_").replace("|", "_").replace(":", "_").replace(" ", "_")

    flow_out = fork_out / "flow_diagrams"
    flow_out.mkdir(parents=True, exist_ok=True)

    for tmeta in active_meta:
        label = tmeta["short_id"]
        domain = tmeta["domain"]
        safe_label = _safe_name(label)

        action_npz = tmeta["npz_path"]
        response_npz = str(Path(response_dir) / Path(action_npz).name)

        act_mat, res_mat, indices = load_paired_embeddings(action_npz, response_npz)
        rewards = _load_rewards_from_files(tmeta["tdp_files"])
        all_rids = sorted(set(r for r, _ in indices) & set(rewards.keys()))
        n_orig = len(all_rids)
        task_sr = sum(1 for r in all_rids if rewards.get(r, 0) >= 0.5) / max(n_orig, 1)

        print(f"\n{'=' * 60}")
        print(f"Task: {label} ({n_orig} rollouts, {act_mat.shape[0]} steps, sr={task_sr:.0%})")
        print(f"{'=' * 60}")

        # -- Baseline flow diagram (original only) --
        S_orig, _, _ = paired_similarity(act_mat, res_mat, "mean")
        if tool_arg_beta > 0:
            orig_tool_info = _extract_step_tool_info(tmeta["tdp_files"], indices)
            S_orig = apply_tool_arg_penalty(S_orig, orig_tool_info, beta=tool_arg_beta)
        orig_labels = hdbscan_cluster_affinity(
            S_orig, min_cluster_size=min_cluster_size,
            noise_strategy=noise_strategy,
        )
        K_orig = int(orig_labels.max()) + 1
        noise_lbl = K_orig - 1 if noise_strategy == "outlier_bin" else None

        baseline_png = flow_out / f"flow_baseline_{safe_label}.png"
        render_trajectory_flow(
            orig_labels, indices, rewards,
            baseline_png,
            title=f"{label} — Baseline (n={n_orig}, K={K_orig}, sr={task_sr:.0%})",
            noise_label=noise_lbl,
        )
        print(f"  Baseline flow: K={K_orig}, saved to {baseline_png.name}")

        # -- Per-strategy: embed forks, merge, re-cluster, plot --
        task_forks = fork_by_task_strat.get(label, {})
        for strat, strat_forks in sorted(task_forks.items()):
            strat_dir = fork_out / strat / domain
            if not strat_dir.exists():
                print(f"  {strat}: no TDP dir at {strat_dir}, skipping")
                continue

            print(f"\n  Strategy: {strat} ({len(strat_forks)} forks for {label})")

            action_texts: list[str] = []
            response_texts: list[str] = []
            fork_indices: list[tuple[int, int]] = []
            fork_rewards: dict[int, float] = {}
            fork_meta_list: list[dict] = []
            fork_tool_info: list[tuple[str, dict] | None] = []

            fork_rid_start = 10000
            for fi, fr in enumerate(strat_forks):
                fork_id = fr["fork_id"]
                tdp_path = strat_dir / f"{fork_id}_tdp.json"
                if not tdp_path.exists():
                    continue

                with open(tdp_path) as f:
                    tdp = json.load(f)

                all_steps = _extract_fork_steps(tdp["messages"])
                if not all_steps:
                    continue

                fork_step = tdp.get("fork_step", 0)
                rid = fork_rid_start + fi
                fork_rewards[rid] = tdp.get("reward", 0.0) or 0.0
                fork_meta_list.append({
                    "rid": rid, "fork_step": fork_step,
                    "total_steps": len(all_steps),
                })

                all_tool_info = _extract_fork_messages_tool_info(tdp["messages"])

                post_fork_start = fork_step + 1
                for si in range(post_fork_start, len(all_steps)):
                    act_text, res_text = all_steps[si]
                    action_texts.append(act_text)
                    response_texts.append(res_text)
                    fork_indices.append((rid, si))
                    if si < len(all_tool_info):
                        fork_tool_info.append(all_tool_info[si])
                    else:
                        fork_tool_info.append(None)

            n_fork_rollouts = len(fork_rewards)
            n_fork_steps = len(action_texts)
            n_prefix_skipped = sum(
                fm["fork_step"] + 1 for fm in fork_meta_list
            )
            print(f"    {n_fork_rollouts} rollouts, "
                  f"{n_fork_steps} post-fork steps to embed "
                  f"({n_prefix_skipped} prefix steps skipped)")

            if not action_texts:
                continue

            cache_path = fork_out / f"embed_cache_{strat}_{safe_label}.npz"
            if cache_path.exists():
                cached = np.load(cache_path)
                fork_act_emb = cached["action"]
                fork_res_emb = cached["response"]
                print(f"    Loaded cached embeddings from {cache_path.name}")
            else:
                print(f"    Embedding actions...")
                fork_act_emb = _embed_texts_sync(action_texts, max_workers=4)
                print(f"    Embedding responses...")
                fork_res_emb = _embed_texts_sync(response_texts, max_workers=4)
                np.savez_compressed(
                    cache_path,
                    action=fork_act_emb, response=fork_res_emb,
                    indices=np.array(fork_indices),
                )
                print(f"    Cached embeddings to {cache_path.name}")

            merged_act = np.vstack([act_mat, fork_act_emb])
            merged_res = np.vstack([res_mat, fork_res_emb])
            merged_indices = list(indices) + fork_indices
            merged_rewards = dict(rewards)
            merged_rewards.update(fork_rewards)

            print(f"    Clustering {merged_act.shape[0]} steps...")
            S_merged, _, _ = paired_similarity(merged_act, merged_res, "mean")
            if tool_arg_beta > 0:
                orig_tool_info_strat = _extract_step_tool_info(tmeta["tdp_files"], indices)
                merged_tool_info = orig_tool_info_strat + fork_tool_info
                S_merged = apply_tool_arg_penalty(S_merged, merged_tool_info, beta=tool_arg_beta)
            merged_labels = hdbscan_cluster_affinity(
                S_merged, min_cluster_size=min_cluster_size,
                noise_strategy=noise_strategy,
            )
            K_merged = int(merged_labels.max()) + 1
            noise_lbl_m = K_merged - 1 if noise_strategy == "outlier_bin" else None

            n_merged = len(set(r for r, _ in merged_indices))
            fork_pass = sum(1 for r in fork_rewards.values() if r >= 0.5)
            fork_fail = n_fork_rollouts - fork_pass
            merged_sr_orig = sum(1 for r in all_rids if rewards.get(r, 0) >= 0.5)
            merged_sr = (merged_sr_orig + fork_pass) / max(n_merged, 1)

            print(f"    Merged: {n_merged} rollouts, "
                  f"K={K_merged}, sr={merged_sr:.0%}")
            print(f"    Fork outcomes: {fork_pass} pass, {fork_fail} fail")

            strat_png = flow_out / f"flow_{strat}_{safe_label}.png"
            render_trajectory_flow(
                merged_labels, merged_indices, merged_rewards,
                strat_png,
                title=(f"{label} — {strat} "
                       f"(n={n_merged}, K={K_merged}, "
                       f"sr={merged_sr:.0%}, +{n_fork_rollouts} forks)"),
                noise_label=noise_lbl_m,
            )
            print(f"    Flow saved: {strat_png.name}")

            npz_path = fork_out / f"embeddings_{strat}_{safe_label}.npz"
            np.savez_compressed(
                npz_path,
                action=merged_act, response=merged_res,
                indices=np.array(merged_indices),
                labels=merged_labels,
            )

    print(f"\nAll flow diagrams saved to {flow_out}")


# ── Fork-augmented PRM training ──────────────────────────────────────

FORK_PRM_REPEATS = 20
FORK_PRM_TRAIN_FRAC = 0.8
FORK_STRATEGIES_ALL = ["minority_weighted", "minority_early", "early_rare"]


def _build_file_index_to_position(tdp_files: list[str]) -> dict[int, int]:
    """Map TDP file numeric index (N in ``_tdp_N.json``) to list position.

    Fork ``parent_rid`` values are file indices assigned at generation time.
    If files have been deleted since then the list positions shift.  This map
    lets us translate a file-index-based parent_rid to the current position
    used by the NPZ embeddings.
    """
    import re as _re
    mapping: dict[int, int] = {}
    for pos, fp in enumerate(tdp_files):
        m = _re.search(r"_tdp_(\d+)\.json$", fp)
        if m:
            mapping[int(m.group(1))] = pos
    return mapping


def _load_fork_data_for_strategy(
    fork_out: Path,
    strat: str,
    strat_results: list[dict],
    domain: str,
    label: str,
    safe_label: str | None = None,
    tdp_files: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], dict[int, float],
           list[tuple[str, dict] | None], dict[int, tuple[int, int]]]:
    """Load fork embeddings + indices + rewards + tool info for one strategy.

    Returns (fork_act, fork_res, fork_indices, fork_rewards, fork_tool_info,
             fork_parent_map) where fork_parent_map maps fork_rid -> (parent_rid, fork_step).
    Reuses embed_cache if available; otherwise re-embeds and caches.

    When *tdp_files* is provided, ``parent_rid`` values in the returned
    ``fork_parent_map`` are translated from file indices to current list
    positions so they match the NPZ rollout IDs.
    """
    if safe_label is None:
        safe_label = label.replace("/", "_").replace("|", "_").replace(":", "_").replace(" ", "_")

    fidx_to_pos = _build_file_index_to_position(tdp_files) if tdp_files else {}

    strat_dir = fork_out / strat / domain
    strat_forks = [r for r in strat_results
                   if label in r.get("task_label", "")]

    action_texts: list[str] = []
    response_texts: list[str] = []
    fork_indices: list[tuple[int, int]] = []
    fork_rewards: dict[int, float] = {}
    fork_tool_info: list[tuple[str, dict] | None] = []
    fork_parent_map: dict[int, tuple[int, int]] = {}
    fork_rid_start = 10000

    for fi, fr in enumerate(strat_forks):
        fork_id = fr["fork_id"]
        tdp_path = strat_dir / f"{fork_id}_tdp.json"
        if not tdp_path.exists():
            continue

        with open(tdp_path) as f:
            tdp = json.load(f)

        all_steps = _extract_fork_steps(tdp["messages"])
        if not all_steps:
            continue

        fork_step = tdp.get("fork_step", 0)
        rid = fork_rid_start + fi
        fork_rewards[rid] = tdp.get("reward", 0.0) or 0.0
        raw_parent = fr["parent_rid"]
        parent_rid = fidx_to_pos.get(raw_parent, raw_parent)
        fork_parent_map[rid] = (parent_rid, fork_step)

        all_ti = _extract_fork_messages_tool_info(tdp["messages"])

        post_fork_start = fork_step + 1
        for si in range(post_fork_start, len(all_steps)):
            act_text, res_text = all_steps[si]
            action_texts.append(act_text)
            response_texts.append(res_text)
            fork_indices.append((rid, si))
            fork_tool_info.append(all_ti[si] if si < len(all_ti) else None)

    cache_path = fork_out / f"embed_cache_{strat}_{safe_label}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        fork_act_emb = cached["action"]
        fork_res_emb = cached["response"]
    else:
        fork_act_emb = _embed_texts_sync(action_texts, max_workers=4)
        fork_res_emb = _embed_texts_sync(response_texts, max_workers=4)
        np.savez_compressed(
            cache_path,
            action=fork_act_emb, response=fork_res_emb,
            indices=np.array(fork_indices),
        )

    return fork_act_emb, fork_res_emb, fork_indices, fork_rewards, fork_tool_info, fork_parent_map


def _run_fork_prm_for_task(
    tmeta: dict,
    fork_out: Path,
    completed: dict[str, list[dict]],
    response_dir: str,
    indices_cache: dict,
    min_cluster_size: int,
    noise_strategy: str,
    tool_arg_beta: float,
    pbar,
    level_filter: set[str] | None = None,
) -> dict | None:
    """Run the fork PRM experiment for a single task. Returns per-task results dict."""
    label = tmeta["short_id"]
    domain = tmeta["domain"]
    safe_label = label.replace("/", "_").replace("|", "_").replace(":", "_").replace(" ", "_")

    action_npz = tmeta["npz_path"]
    response_npz = str(Path(response_dir) / Path(action_npz).name)

    act_mat, res_mat, indices = load_paired_embeddings(action_npz, response_npz)
    rewards = _load_rewards_from_files(tmeta["tdp_files"])
    orig_rids = sorted(set(r for r, _ in indices) & set(rewards.keys()))
    n_orig = len(orig_rids)
    task_sr = sum(1 for r in orig_rids if rewards.get(r, 0) >= 0.5) / max(n_orig, 1)

    print(f"\n{'=' * 60}")
    print(f"Fork PRM: {label} ({n_orig} rollouts, sr={task_sr:.0%})")
    print(f"{'=' * 60}")

    strat_fork_data: dict[str, tuple] = {}
    for strat in FORK_STRATEGIES_ALL:
        if strat not in completed:
            continue
        strat_dir = fork_out / strat / domain
        if not strat_dir.exists():
            continue

        fork_act, fork_res, fork_idx, fork_rew, fork_ti, fork_pmap = \
            _load_fork_data_for_strategy(
                fork_out, strat, completed[strat], domain, label,
                safe_label=safe_label, tdp_files=tmeta["tdp_files"],
            )
        n_forks = len(fork_rew)
        if n_forks == 0:
            continue
        fork_sr = sum(1 for v in fork_rew.values() if v >= 0.5) / n_forks
        print(f"  {strat}: {n_forks} forks, {fork_act.shape[0]} post-fork steps, "
              f"fork_sr={fork_sr:.2f}")
        strat_fork_data[strat] = (fork_act, fork_res, fork_idx, fork_rew, fork_ti, fork_pmap)

    if not strat_fork_data:
        print(f"  No fork data found for {label}, skipping")
        return None

    conditions = ["baseline"] + [s for s in FORK_STRATEGIES_ALL if s in strat_fork_data]

    cond_step_emb: dict[str, np.ndarray] = {}
    cond_rid_to_rows: dict[str, dict[int, list[int]]] = {}

    orig_step_emb = np.hstack([act_mat, res_mat])
    orig_rid_to_rows: dict[int, list[int]] = {}
    for i, (rid, _step) in enumerate(indices):
        orig_rid_to_rows.setdefault(rid, []).append(i)
    cond_step_emb["baseline"] = orig_step_emb
    cond_rid_to_rows["baseline"] = orig_rid_to_rows

    for strat, (fork_act, fork_res, fork_idx, fork_rew, _fork_ti, _fork_pm) in strat_fork_data.items():
        full_act = np.vstack([act_mat, fork_act])
        full_res = np.vstack([res_mat, fork_res])
        full_step_emb = np.hstack([full_act, full_res])
        full_indices = list(indices) + list(fork_idx)
        r2r: dict[int, list[int]] = {}
        for i, (rid, _step) in enumerate(full_indices):
            r2r.setdefault(rid, []).append(i)
        cond_step_emb[strat] = full_step_emb
        cond_rid_to_rows[strat] = r2r

    cond_levels: dict[str, dict] = {}
    for cond in conditions:
        lvls = dict(LEVELS)
        emb_args = {
            "step_embeddings": cond_step_emb[cond],
            "rid_to_rows": cond_rid_to_rows[cond],
        }
        lvls["causal_emb_sim"] = (CausalEmbSimBaseline, dict(emb_args))
        lvls["causal_knn"] = (CausalKnnBaseline, {**emb_args, "k": 5})
        if level_filter:
            lvls = {k: v for k, v in lvls.items() if k in level_filter}
        cond_levels[cond] = lvls

    all_level_names = list(cond_levels[conditions[0]].keys())
    n_train = int(n_orig * FORK_PRM_TRAIN_FRAC)

    all_metrics: dict[str, dict[str, dict[str, dict]]] = {}
    for cond in conditions:
        all_metrics[cond] = {}
        for lname in all_level_names:
            all_metrics[cond][lname] = {
                agg: {"auroc": [], "bon": {N: [] for N in N_VALUES}}
                for agg in STRATEGIES
            }

    orig_tool_info: list[tuple[str, dict] | None] | None = None
    if tool_arg_beta > 0:
        orig_tool_info = _extract_step_tool_info(tmeta["tdp_files"], indices)

    for rep in range(FORK_PRM_REPEATS):
        rng_split = np.random.default_rng(rep * 1000 + 42)
        perm = rng_split.permutation(orig_rids)
        train_orig_rids = sorted(perm[:n_train])
        test_rids = sorted(perm[n_train:])
        train_orig_set = set(train_orig_rids)
        test_set = set(test_rids)

        train_orig_mask = np.array(
            [i for i, (r, _) in enumerate(indices) if r in train_orig_set])
        test_mask = np.array(
            [i for i, (r, _) in enumerate(indices) if r in test_set])

        test_act = act_mat[test_mask]
        test_res = res_mat[test_mask]
        test_indices = [indices[i] for i in test_mask]

        ho_outcomes = {r: float(rewards.get(r, 0) >= 0.5) for r in test_rids}

        for cond in conditions:
            sample_weights: dict[int, float] | None = None
            fork_pmap: dict[int, tuple[int, int]] = {}
            if cond == "baseline":
                train_act = act_mat[train_orig_mask]
                train_res = res_mat[train_orig_mask]
                train_indices = [indices[i] for i in train_orig_mask]
                train_rewards = {r: rewards[r] for r in train_orig_rids}
            else:
                fork_act, fork_res, fork_idx, fork_rew, fork_ti, fork_pmap = \
                    strat_fork_data[cond]
                train_act = np.vstack([act_mat[train_orig_mask], fork_act])
                train_res = np.vstack([res_mat[train_orig_mask], fork_res])
                train_indices = [indices[i] for i in train_orig_mask] + list(fork_idx)
                train_rewards = {r: rewards[r] for r in train_orig_rids}
                train_rewards.update(fork_rew)

            S_train, _, _ = paired_similarity(train_act, train_res, "mean")
            if tool_arg_beta > 0 and orig_tool_info is not None:
                train_ti = [orig_tool_info[i] for i in train_orig_mask]
                if cond != "baseline":
                    train_ti = train_ti + fork_ti  # type: ignore[possibly-undefined]
                S_train = apply_tool_arg_penalty(S_train, train_ti, beta=tool_arg_beta)
            effective_mcs = max(2, min(min_cluster_size, len(train_act) // 5))
            train_labels = hdbscan_cluster_affinity(
                S_train, min_cluster_size=effective_mcs,
                noise_strategy=noise_strategy,
            )
            K = int(train_labels.max()) + 1

            _, train_rollout_paths = build_global_thought_graph(
                train_labels, train_indices, train_rewards,
            )
            train_path_rids = sorted(
                set(train_rollout_paths.keys()) & set(train_rewards.keys()))
            train_rollout_paths = {r: train_rollout_paths[r] for r in train_path_rids}
            train_rew_filtered = {r: train_rewards[r] for r in train_path_rids}

            # ── Full-path reconstruction + sample weighting for forks ──
            if cond != "baseline" and fork_pmap:
                parent_child_count: dict[int, int] = {}
                n_usable = 0
                for frid, (prid, _fstep) in fork_pmap.items():
                    if frid in train_rollout_paths and prid in train_rollout_paths:
                        parent_child_count[prid] = parent_child_count.get(prid, 0) + 1
                        n_usable += 1

                drop_frids = []
                for frid, (prid, fstep) in fork_pmap.items():
                    if frid not in train_rollout_paths:
                        continue
                    if prid not in train_rollout_paths:
                        drop_frids.append(frid)
                        continue
                    parent_path = train_rollout_paths[prid]
                    prefix_len = min(fstep + 1, len(parent_path))
                    fork_suffix = train_rollout_paths[frid]
                    train_rollout_paths[frid] = parent_path[:prefix_len] + fork_suffix

                for frid in drop_frids:
                    train_rollout_paths.pop(frid, None)
                    train_rew_filtered.pop(frid, None)

                n_train_orig = len(train_orig_set & set(train_rollout_paths.keys()))
                n_total = n_train_orig + n_usable
                sample_weights = {}
                for rid in train_rollout_paths:
                    if rid in train_orig_set:
                        sample_weights[rid] = 1.0
                    elif rid in fork_pmap:
                        prid = fork_pmap[rid][0]
                        n_sibs = parent_child_count.get(prid, 1)
                        sample_weights[rid] = (n_train_orig / max(n_total, 1)) / n_sibs
                    else:
                        sample_weights[rid] = 1.0

            ho_labels = _assign_heldout_to_clusters(
                train_act, train_res, train_labels, test_act, test_res,
            )
            _, ho_rollout_paths = build_global_thought_graph(
                ho_labels, test_indices, rewards,
            )
            ho_eval_rids = sorted(
                set(ho_rollout_paths.keys()) & set(rewards.keys()) & test_set)

            active_levels = cond_levels[cond]
            for level_name, (prm_class, prm_kwargs) in active_levels.items():
                prm = prm_class(**prm_kwargs)
                prm.fit(train_rollout_paths, train_rew_filtered, K,
                        sample_weights=sample_weights)

                step_scores: dict[int, list[float]] = {}
                for r in ho_eval_rids:
                    if r in ho_rollout_paths:
                        step_scores[r] = prm.score_path(ho_rollout_paths[r], rid=r)

                scored_rids = [r for r in ho_eval_rids if r in step_scores]
                if len(scored_rids) < 3:
                    pbar.update(1)
                    continue

                rng_eval = np.random.default_rng(rep * 100_000 + hash(cond) % 10000 + 7)
                for agg_strat in STRATEGIES:
                    agg = {r: _aggregate_scores(step_scores[r], agg_strat)
                           for r in scored_rids}
                    auroc_val = _evaluate_auroc(scored_rids, ho_outcomes, agg)
                    bon_vals = _evaluate_bon(
                        scored_rids, ho_outcomes, agg, N_VALUES, rng_eval)

                    bucket = all_metrics[cond][level_name][agg_strat]
                    bucket["auroc"].append(auroc_val)
                    for N in N_VALUES:
                        bucket["bon"][N].append(bon_vals[N])

                pbar.update(1)

    sr = task_sr
    task_result: dict = {
        "task": label, "n_orig": n_orig, "sr": sr,
        "n_repeats": FORK_PRM_REPEATS,
        "train_size": n_train, "test_size": n_orig - n_train,
        "conditions": {},
    }

    for cond in conditions:
        cond_d: dict = {"levels": {}}
        for lname in all_level_names:
            level_d: dict = {"strategies": {}}
            for agg_strat in STRATEGIES:
                bucket = all_metrics[cond][lname][agg_strat]
                auroc_vals = bucket["auroc"]
                strat_d: dict = {
                    "auroc_mean": float(np.mean(auroc_vals)) if auroc_vals else 0.5,
                    "auroc_std": float(np.std(auroc_vals)) if auroc_vals else 0.0,
                    "bon": {},
                }
                for N in N_VALUES:
                    bv = bucket["bon"][N]
                    strat_d["bon"][str(N)] = {
                        "mean": float(np.mean(bv)) if bv else sr,
                        "std": float(np.std(bv)) if bv else 0.0,
                        "random": sr,
                        "oracle": 1 - (1 - sr) ** N,
                    }
                level_d["strategies"][agg_strat] = strat_d
            cond_d["levels"][lname] = level_d
        n_forks = len(strat_fork_data.get(cond, ([], [], [], {}, [], {}))[3]) if cond != "baseline" else 0
        cond_d["n_train_rollouts"] = n_train + n_forks
        task_result["conditions"][cond] = cond_d

    # Print per-task summary
    print(f"\n  {'Condition':<22s} {'Level':<22s} {'AUROC':>7s}  {'BoN@8':>7s}")
    print(f"  {'-' * 62}")
    for cond in conditions:
        for lname in all_level_names:
            sd = task_result["conditions"][cond]["levels"][lname]["strategies"]["final"]
            auroc = sd["auroc_mean"]
            bon8 = sd["bon"]["8"]["mean"]
            print(f"  {cond:<22s} {lname:<22s} {auroc:7.3f}  {bon8:7.3f}")

    return task_result


def run_fork_prm_experiment(
    data_dir: str,
    out_dir: str,
    min_cluster_size: int = 5,
    noise_strategy: str = "outlier_bin",
    tool_arg_beta: float = 0.0,
    level_filter: set[str] | None = None,
) -> dict:
    """Train PRMs on fork-augmented data for all tasks, evaluate on held-out originals.

    For each task with fork data, runs FORK_PRM_REPEATS random 80/20 splits
    of the original rollouts, trains PRMs under four conditions
    (baseline + 3 fork strategies), and evaluates AUROC/BoN on held-out originals.
    """
    import warnings
    from tqdm import tqdm

    warnings.filterwarnings("ignore", category=FutureWarning)

    fork_out = Path(out_dir) / "minority_fork"
    results_jsonl = fork_out / "fork_results.jsonl"
    if not results_jsonl.exists():
        print("No fork results found. Run --fork first.")
        return {}

    completed: dict[str, list[dict]] = {}
    for line in open(results_jsonl):
        r = json.loads(line)
        if r.get("error"):
            continue
        completed.setdefault(r.get("strategy", "unknown"), []).append(r)

    fork_task_labels = set()
    for strat_results in completed.values():
        for r in strat_results:
            fork_task_labels.add(r.get("task_label", ""))

    base = Path(data_dir)
    action_dir = str(base / "embeddings" / "action")
    response_dir = str(base / "embeddings" / "response")

    task_meta = _discover_tasks(data_dir, action_dir)
    active_meta = [t for t in task_meta if t["short_id"] in fork_task_labels]
    if not active_meta:
        print(f"No task metadata matches fork labels: {fork_task_labels}")
        return {}

    print(f"Fork PRM experiment across {len(active_meta)} tasks: "
          f"{[t['short_id'] for t in active_meta]}")

    if level_filter:
        n_levels = len(level_filter)
    else:
        n_levels = len(LEVELS) + 2  # +2 for embedding baselines
    total_iters = (
        len(active_meta)
        * FORK_PRM_REPEATS
        * (1 + len(FORK_STRATEGIES_ALL))
        * n_levels
    )

    pbar = tqdm(total=total_iters, desc="Fork PRM (all tasks)", unit="run")

    all_task_results: dict[str, dict] = {}
    for tmeta in active_meta:
        task_result = _run_fork_prm_for_task(
            tmeta, fork_out, completed, response_dir, {},
            min_cluster_size, noise_strategy, tool_arg_beta, pbar,
            level_filter=level_filter,
        )
        if task_result is not None:
            all_task_results[tmeta["short_id"]] = task_result

    pbar.close()

    combined = {"tasks": all_task_results}
    results_path = fork_out / "fork_prm_results.json"
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nAll results saved to {results_path}")

    plot_fork_prm_comparison_multi(all_task_results, fork_out)
    return combined


def run_fork_loo_experiment(
    data_dir: str,
    out_dir: str,
    fork_data_dir: str | None = None,
    min_cluster_size: int = 5,
    noise_strategy: str = "outlier_bin",
    tool_arg_beta: float = 0.0,
    level_filter: set[str] | None = None,
    recluster: bool = False,
) -> dict:
    """LOO-CV on original rollouts with fork-augmented training sets.

    For each held-out original rollout, trains PRMs on the remaining originals
    plus cold-fork data (per strategy).  Two clustering modes:

    * ``recluster=False`` (default): cluster on original train embeddings,
      NN-assign fork embeddings to those clusters.
    * ``recluster=True``: re-cluster on combined original-train + fork embeddings.

    Conditions: *baseline* (no forks) and one per fork strategy.

    *fork_data_dir* points to the directory with fork strategy subdirs,
    embed caches, and ``fork_results.jsonl``.  Defaults to
    ``out_dir / "minority_fork"`` for backward compat.
    """
    import warnings
    from tqdm import tqdm

    warnings.filterwarnings("ignore", category=FutureWarning)

    fork_out = Path(fork_data_dir) if fork_data_dir else Path(out_dir) / "minority_fork"

    results_jsonl = fork_out / "fork_results.jsonl"
    if not results_jsonl.exists():
        for candidate in [fork_out.parent / "fork_results.jsonl",
                          fork_out.parent / "results" / "fork_results.jsonl"]:
            if candidate.exists():
                results_jsonl = candidate
                break
    if not results_jsonl.exists():
        print(f"No fork_results.jsonl found (searched {fork_out} and parent). "
              "Run --fork first.")
        return {}

    completed: dict[str, list[dict]] = {}
    for line in open(results_jsonl):
        r = json.loads(line)
        if r.get("error"):
            continue
        completed.setdefault(r.get("strategy", "unknown"), []).append(r)

    fork_task_labels = set()
    for strat_results in completed.values():
        for r in strat_results:
            fork_task_labels.add(r.get("task_label", ""))

    base = Path(data_dir)
    action_dir = str(base / "embeddings" / "action")
    response_dir = str(base / "embeddings" / "response")

    task_meta = _discover_tasks(data_dir, action_dir)
    active_meta = [t for t in task_meta if t["short_id"] in fork_task_labels]
    if not active_meta:
        print(f"No task metadata matches fork labels: {fork_task_labels}")
        return {}

    cluster_mode = "recluster" if recluster else "nn_assign"
    print(f"Fork LOO experiment ({cluster_mode}) across {len(active_meta)} tasks")

    results_dir = Path(out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_task_results: dict[str, dict] = {}

    for tmeta in active_meta:
        label = tmeta["short_id"]
        domain = tmeta["domain"]
        safe_label = label.replace("/", "_").replace("|", "_").replace(":", "_").replace(" ", "_")

        action_npz = tmeta["npz_path"]
        response_npz = str(Path(response_dir) / Path(action_npz).name)

        act_mat, res_mat, indices = load_paired_embeddings(action_npz, response_npz)
        rewards = _load_rewards_from_files(tmeta["tdp_files"])
        orig_rids = sorted(set(r for r, _ in indices) & set(rewards.keys()))
        n_orig = len(orig_rids)
        task_sr = sum(1 for r in orig_rids if rewards.get(r, 0) >= 0.5) / max(n_orig, 1)

        strat_fork_data: dict[str, tuple] = {}
        for strat in FORK_STRATEGIES_ALL:
            if strat not in completed:
                continue
            strat_dir = fork_out / strat / domain
            if not strat_dir.exists():
                continue
            fork_act, fork_res, fork_idx, fork_rew, fork_ti, fork_pmap = \
                _load_fork_data_for_strategy(
                    fork_out, strat, completed[strat], domain, label,
                    safe_label=safe_label, tdp_files=tmeta["tdp_files"],
                )
            if len(fork_rew) == 0:
                continue
            fork_sr = sum(1 for v in fork_rew.values() if v >= 0.5) / len(fork_rew)
            strat_fork_data[strat] = (fork_act, fork_res, fork_idx, fork_rew, fork_ti, fork_pmap)
            print(f"  {strat}: {len(fork_rew)} forks, sr={fork_sr:.2f}")

        conditions = ["baseline"] + [s for s in FORK_STRATEGIES_ALL if s in strat_fork_data]

        print(f"\n{'=' * 60}")
        print(f"Fork LOO ({cluster_mode}): {label}")
        print(f"  {n_orig} originals, sr={task_sr:.0%}")
        print(f"  conditions: {conditions}")
        print(f"{'=' * 60}")

        orig_tool_info: list[tuple[str, dict] | None] | None = None
        if tool_arg_beta > 0:
            orig_tool_info = _extract_step_tool_info(tmeta["tdp_files"], indices)

        active_levels = dict(LEVELS)
        if level_filter:
            active_levels = {k: v for k, v in active_levels.items()
                            if k in level_filter}
        level_names = list(active_levels.keys())

        # cond_scores[level][cond][rid] = aggregated score
        cond_scores: dict[str, dict[str, dict[int, float]]] = {
            lname: {c: {} for c in conditions} for lname in level_names
        }

        for hold_idx, held_out_rid in enumerate(tqdm(
            orig_rids, desc=f"LOO {_short_task_label(label)}", leave=True,
        )):
            train_rids = [r for r in orig_rids if r != held_out_rid]
            train_set = set(train_rids)

            train_mask = np.array(
                [i for i, (r, _) in enumerate(indices) if r in train_set])
            ho_mask = np.array(
                [i for i, (r, _) in enumerate(indices) if r == held_out_rid])

            if len(ho_mask) == 0:
                continue

            ho_act = act_mat[ho_mask]
            ho_res = res_mat[ho_mask]
            ho_indices = [indices[i] for i in ho_mask]

            for cond in conditions:
                fork_pmap: dict[int, tuple[int, int]] = {}

                if cond == "baseline":
                    train_act = act_mat[train_mask]
                    train_res = res_mat[train_mask]
                    train_indices = [indices[i] for i in train_mask]
                    train_rewards = {r: rewards[r] for r in train_rids}
                else:
                    fork_act, fork_res, fork_idx, fork_rew, fork_ti, fork_pmap = \
                        strat_fork_data[cond]
                    if recluster:
                        train_act = np.vstack([act_mat[train_mask], fork_act])
                        train_res = np.vstack([res_mat[train_mask], fork_res])
                        train_indices = [indices[i] for i in train_mask] + list(fork_idx)
                    else:
                        train_act = act_mat[train_mask]
                        train_res = res_mat[train_mask]
                        train_indices = [indices[i] for i in train_mask]
                    train_rewards = {r: rewards[r] for r in train_rids}
                    train_rewards.update(fork_rew)

                S_train, _, _ = paired_similarity(train_act, train_res, "mean")
                if tool_arg_beta > 0 and orig_tool_info is not None:
                    train_ti = [orig_tool_info[i] for i in train_mask]
                    if cond != "baseline" and recluster:
                        train_ti = train_ti + fork_ti  # type: ignore[possibly-undefined]
                    S_train = apply_tool_arg_penalty(S_train, train_ti, beta=tool_arg_beta)
                effective_mcs = max(2, min(min_cluster_size, len(train_act) // 5))
                train_labels = hdbscan_cluster_affinity(
                    S_train, min_cluster_size=effective_mcs,
                    noise_strategy=noise_strategy,
                )
                K = int(train_labels.max()) + 1

                _, train_rollout_paths = build_global_thought_graph(
                    train_labels, train_indices, train_rewards,
                )
                train_rollout_paths = {
                    r: train_rollout_paths[r]
                    for r in train_rollout_paths
                    if r in train_rewards
                }
                train_rew_filtered = {r: train_rewards[r] for r in train_rollout_paths}

                if cond != "baseline" and not recluster and fork_pmap:
                    fork_act_c, fork_res_c = strat_fork_data[cond][:2]
                    fork_idx_c = strat_fork_data[cond][2]
                    fork_labels = _assign_heldout_to_clusters(
                        train_act, train_res, train_labels,
                        fork_act_c, fork_res_c,
                    )
                    _, fork_paths = build_global_thought_graph(
                        fork_labels, fork_idx_c, train_rewards,
                    )
                    train_rollout_paths.update(fork_paths)
                    train_rew_filtered = {r: train_rewards[r] for r in train_rollout_paths}

                sample_weights: dict[int, float] | None = None
                if cond != "baseline" and fork_pmap:
                    parent_child_count: dict[int, int] = {}
                    n_usable = 0
                    for frid, (prid, _fstep) in fork_pmap.items():
                        if frid in train_rollout_paths and prid in train_rollout_paths:
                            parent_child_count[prid] = parent_child_count.get(prid, 0) + 1
                            n_usable += 1

                    drop_frids = []
                    for frid, (prid, fstep) in fork_pmap.items():
                        if frid not in train_rollout_paths:
                            continue
                        if prid not in train_rollout_paths:
                            drop_frids.append(frid)
                            continue
                        parent_path = train_rollout_paths[prid]
                        prefix_len = min(fstep + 1, len(parent_path))
                        fork_suffix = train_rollout_paths[frid]
                        train_rollout_paths[frid] = parent_path[:prefix_len] + fork_suffix

                    for frid in drop_frids:
                        train_rollout_paths.pop(frid, None)
                        train_rew_filtered.pop(frid, None)

                    n_train_orig = len(train_set & set(train_rollout_paths.keys()))
                    n_total = n_train_orig + n_usable
                    sample_weights = {}
                    for rid in train_rollout_paths:
                        if rid in train_set:
                            sample_weights[rid] = 1.0
                        elif rid in fork_pmap:
                            prid = fork_pmap[rid][0]
                            n_sibs = parent_child_count.get(prid, 1)
                            sample_weights[rid] = (n_train_orig / max(n_total, 1)) / n_sibs
                        else:
                            sample_weights[rid] = 1.0

                ho_labels = _assign_heldout_to_clusters(
                    train_act, train_res, train_labels, ho_act, ho_res,
                )
                _, ho_paths = build_global_thought_graph(
                    ho_labels, ho_indices, rewards,
                )

                if held_out_rid not in ho_paths:
                    continue

                for lname, (prm_class, prm_kwargs) in active_levels.items():
                    prm = prm_class(**prm_kwargs)
                    prm.fit(train_rollout_paths, train_rew_filtered, K,
                            sample_weights=sample_weights)
                    step_sc = prm.score_path(ho_paths[held_out_rid], rid=held_out_rid)
                    cond_scores[lname][cond][held_out_rid] = \
                        _aggregate_scores(step_sc, "final")

        outcomes = {r: float(rewards.get(r, 0) >= 0.5) for r in orig_rids}
        rng = np.random.default_rng(42)

        task_result: dict = {
            "task": label, "n_orig": n_orig, "sr": task_sr,
            "cluster_mode": cluster_mode,
            "levels": {},
        }

        for lname in level_names:
            level_d: dict = {"conditions": {}}

            print(f"\n  {lname}:")
            print(f"    {'Condition':<25s} {'AUROC':>7s}  {'BoN@8':>7s}")
            print(f"    {'-'*45}")

            for cond in conditions:
                sc = cond_scores[lname][cond]
                scored_rids = sorted(sc.keys())
                if len(scored_rids) < 3:
                    continue
                auroc = _evaluate_auroc(scored_rids, outcomes, sc)
                bon = _evaluate_bon(scored_rids, outcomes, sc, N_VALUES, rng)

                cond_d: dict = {
                    "auroc": auroc,
                    "n_scored": len(scored_rids),
                    "bon": {},
                }
                for N in N_VALUES:
                    cond_d["bon"][N] = {
                        "value": bon[N],
                        "random": task_sr,
                        "oracle": 1 - (1 - task_sr) ** N,
                    }
                level_d["conditions"][cond] = cond_d
                print(f"    {cond:<25s} {auroc:7.3f}  {bon.get(8, 0):7.3f}")

            task_result["levels"][lname] = level_d

        all_task_results[label] = task_result

    combined = {"cluster_mode": cluster_mode, "tasks": all_task_results}
    results_path = results_dir / "fork_loo_results.json"
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return combined


# ── Fork + heldout / combined-LOO experiments ────────────────────────


def _load_fork_completed(fork_data_dir: Path) -> dict[str, list[dict]]:
    """Load fork_results.jsonl from *fork_data_dir* or its parent/results."""
    candidates = [
        fork_data_dir / "fork_results.jsonl",
        fork_data_dir.parent / "fork_results.jsonl",
        fork_data_dir.parent / "results" / "fork_results.jsonl",
    ]
    for path in candidates:
        if path.exists():
            completed: dict[str, list[dict]] = {}
            for line in open(path):
                r = json.loads(line)
                if r.get("error"):
                    continue
                completed.setdefault(r.get("strategy", "unknown"), []).append(r)
            return completed
    raise FileNotFoundError(
        f"fork_results.jsonl not found (searched {[str(c) for c in candidates]})"
    )


def run_fork_heldout_eval(
    data_dir: str,
    heldout_dir: str,
    out_dir: str,
    fork_data_dir: str | None = None,
    min_cluster_size: int = 5,
    noise_strategy: str = "outlier_bin",
    tool_arg_beta: float = 0.0,
    level_filter: set[str] | None = None,
    recluster: bool = False,
) -> dict:
    """Train PRMs on original data + cold forks, evaluate on held-out rollouts.

    For each task with fork data, compares baseline (originals only) against
    fork-augmented conditions.  Two clustering modes via *recluster*.
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    fork_out = Path(fork_data_dir) if fork_data_dir else Path(out_dir) / "minority_fork"
    completed = _load_fork_completed(fork_out)

    train_tasks = load_all_tasks(
        data_dir, min_cluster_size=min_cluster_size,
        noise_strategy=noise_strategy, tool_arg_beta=tool_arg_beta,
    )

    base = Path(data_dir)
    train_task_meta = _discover_tasks(
        data_dir, str(base / "embeddings" / "action"))

    cluster_mode = "recluster" if recluster else "nn_assign"
    results_dir = Path(out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    active_levels = dict(LEVELS)
    if level_filter:
        active_levels = {k: v for k, v in active_levels.items()
                        if k in level_filter}
    level_names = list(active_levels.keys())

    all_results: dict[str, dict] = {}
    rng = np.random.default_rng(42)

    for train_task in train_tasks:
        tmeta = next(
            (t for t in train_task_meta if t["short_id"] == train_task.label),
            None,
        )
        if tmeta is None:
            continue

        ho = _load_heldout_for_task(train_task, heldout_dir)
        if ho is None:
            print(f"\n  Skipping {train_task.label} — no heldout data")
            continue

        label = train_task.label
        domain = train_task.domain
        safe_label = label.replace("/", "_").replace("|", "_").replace(":", "_").replace(" ", "_")

        strat_fork_data: dict[str, tuple] = {}
        for strat in FORK_STRATEGIES_ALL:
            if strat not in completed:
                continue
            strat_dir = fork_out / strat / domain
            if not strat_dir.exists():
                continue
            fork_act, fork_res, fork_idx, fork_rew, fork_ti, fork_pmap = \
                _load_fork_data_for_strategy(
                    fork_out, strat, completed[strat], domain, label,
                    safe_label=safe_label, tdp_files=tmeta["tdp_files"],
                )
            if len(fork_rew) == 0:
                continue
            strat_fork_data[strat] = (fork_act, fork_res, fork_idx, fork_rew, fork_ti, fork_pmap)

        conditions = ["baseline"] + [s for s in FORK_STRATEGIES_ALL if s in strat_fork_data]

        n_ho = len(ho["all_rids"])
        sr_train = float(np.mean(train_task.outcome))
        sr_ho = int(ho["outcome"].sum()) / n_ho if n_ho else 0

        print(f"\n{'#' * 60}")
        print(f"# Fork heldout ({cluster_mode}): {label}")
        print(f"#   Train: {len(train_task.all_rids)} rollouts, SR={sr_train:.0%}")
        print(f"#   Test:  {n_ho} rollouts, SR={sr_ho:.0%}")
        print(f"#   Conditions: {conditions}")
        print(f"{'#' * 60}")

        dim = train_task.step_embeddings.shape[1] // 2
        orig_act = train_task.step_embeddings[:, :dim]
        orig_res = train_task.step_embeddings[:, dim:]
        orig_indices = train_task.indices

        task_result: dict = {
            "task": label, "mode": "fork_heldout",
            "cluster_mode": cluster_mode,
            "n_train": len(train_task.all_rids), "n_test": n_ho,
            "sr_train": sr_train, "sr_test": sr_ho,
            "levels": {},
        }

        for cond in conditions:
            fork_pmap: dict[int, tuple[int, int]] = {}

            if cond == "baseline":
                train_act = orig_act
                train_res = orig_res
                train_indices = list(orig_indices)
                train_rewards = dict(train_task.rewards)
            else:
                fork_act, fork_res, fork_idx, fork_rew, fork_ti, fork_pmap = \
                    strat_fork_data[cond]
                if recluster:
                    train_act = np.vstack([orig_act, fork_act])
                    train_res = np.vstack([orig_res, fork_res])
                    train_indices = list(orig_indices) + list(fork_idx)
                else:
                    train_act = orig_act
                    train_res = orig_res
                    train_indices = list(orig_indices)
                train_rewards = dict(train_task.rewards)
                train_rewards.update(fork_rew)

            S_train, _, _ = paired_similarity(train_act, train_res, "mean")
            effective_mcs = max(2, min(min_cluster_size, len(train_act) // 5))
            train_labels = hdbscan_cluster_affinity(
                S_train, min_cluster_size=effective_mcs,
                noise_strategy=noise_strategy,
            )
            K = int(train_labels.max()) + 1

            _, train_rollout_paths = build_global_thought_graph(
                train_labels, train_indices, train_rewards,
            )
            train_rollout_paths = {
                r: train_rollout_paths[r]
                for r in train_rollout_paths if r in train_rewards
            }
            train_rew_filtered = {r: train_rewards[r] for r in train_rollout_paths}

            if cond != "baseline" and not recluster and fork_pmap:
                fork_act_c, fork_res_c = strat_fork_data[cond][:2]
                fork_idx_c = strat_fork_data[cond][2]
                fork_labels = _assign_heldout_to_clusters(
                    train_act, train_res, train_labels,
                    fork_act_c, fork_res_c,
                )
                _, fork_paths = build_global_thought_graph(
                    fork_labels, fork_idx_c, train_rewards,
                )
                train_rollout_paths.update(fork_paths)
                train_rew_filtered = {r: train_rewards[r] for r in train_rollout_paths}

            sample_weights: dict[int, float] | None = None
            if cond != "baseline" and fork_pmap:
                orig_set = set(train_task.all_rids)
                parent_child_count: dict[int, int] = {}
                n_usable = 0
                for frid, (prid, _fs) in fork_pmap.items():
                    if frid in train_rollout_paths and prid in train_rollout_paths:
                        parent_child_count[prid] = parent_child_count.get(prid, 0) + 1
                        n_usable += 1
                drop_frids = []
                for frid, (prid, fstep) in fork_pmap.items():
                    if frid not in train_rollout_paths:
                        continue
                    if prid not in train_rollout_paths:
                        drop_frids.append(frid)
                        continue
                    parent_path = train_rollout_paths[prid]
                    prefix_len = min(fstep + 1, len(parent_path))
                    fork_suffix = train_rollout_paths[frid]
                    train_rollout_paths[frid] = parent_path[:prefix_len] + fork_suffix
                for frid in drop_frids:
                    train_rollout_paths.pop(frid, None)
                    train_rew_filtered.pop(frid, None)
                n_train_orig = len(orig_set & set(train_rollout_paths.keys()))
                n_total = n_train_orig + n_usable
                sample_weights = {}
                for rid in train_rollout_paths:
                    if rid in orig_set:
                        sample_weights[rid] = 1.0
                    elif rid in fork_pmap:
                        prid = fork_pmap[rid][0]
                        n_sibs = parent_child_count.get(prid, 1)
                        sample_weights[rid] = (n_train_orig / max(n_total, 1)) / n_sibs
                    else:
                        sample_weights[rid] = 1.0

            ho_outcomes = {r: float(ho["rewards"].get(r, 0) >= 0.5)
                          for r in ho["all_rids"]}

            for lname, (prm_class, prm_kwargs) in active_levels.items():
                prm = prm_class(**prm_kwargs)
                prm.fit(train_rollout_paths, train_rew_filtered, K,
                        sample_weights=sample_weights)

                step_scores: dict[int, list[float]] = {}
                for r in ho["all_rids"]:
                    step_scores[r] = prm.score_path(ho["rollout_paths"][r], rid=r)

                for strat in STRATEGIES:
                    agg = {r: _aggregate_scores(step_scores[r], strat)
                           for r in ho["all_rids"]}
                    auroc = _evaluate_auroc(ho["all_rids"], ho_outcomes, agg)
                    bon = _evaluate_bon(ho["all_rids"], ho_outcomes, agg,
                                        N_VALUES, rng)

                    key = f"{lname}__{cond}__{strat}"
                    task_result.setdefault("results", {})[key] = {
                        "level": lname, "condition": cond,
                        "strategy": strat, "auroc": auroc,
                        "bon": {N: bon[N] for N in N_VALUES},
                    }

                auroc_final = _evaluate_auroc(
                    ho["all_rids"], ho_outcomes,
                    {r: _aggregate_scores(step_scores[r], "final")
                     for r in ho["all_rids"]},
                )
                bon8 = _evaluate_bon(
                    ho["all_rids"], ho_outcomes,
                    {r: _aggregate_scores(step_scores[r], "final")
                     for r in ho["all_rids"]},
                    N_VALUES, rng,
                ).get(8, 0)
                print(f"  {cond:<22s} {lname:<22s} AUROC={auroc_final:.3f}  BoN@8={bon8:.3f}")

        all_results[label] = task_result

    out_path = results_dir / "fork_heldout_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return all_results


def run_fork_combined_loo(
    data_dir: str,
    heldout_dir: str,
    out_dir: str,
    fork_data_dir: str | None = None,
    min_cluster_size: int = 5,
    noise_strategy: str = "outlier_bin",
    tool_arg_beta: float = 0.0,
    level_filter: set[str] | None = None,
    recluster: bool = False,
) -> dict:
    """LOO-CV on combined original + heldout rollouts, with fork-augmented training.

    For each held-out rollout from the combined pool (~300), the training set
    includes the remaining ~299 originals/heldout PLUS all cold-fork data.
    """
    import warnings
    from tqdm import tqdm

    warnings.filterwarnings("ignore", category=FutureWarning)

    fork_out = Path(fork_data_dir) if fork_data_dir else Path(out_dir) / "minority_fork"
    completed = _load_fork_completed(fork_out)

    train_tasks = load_all_tasks(
        data_dir, min_cluster_size=min_cluster_size,
        noise_strategy=noise_strategy, tool_arg_beta=tool_arg_beta,
    )

    base = Path(data_dir)
    train_task_meta = _discover_tasks(
        data_dir, str(base / "embeddings" / "action"))

    cluster_mode = "recluster" if recluster else "nn_assign"
    results_dir = Path(out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    active_levels = dict(LEVELS)
    if level_filter:
        active_levels = {k: v for k, v in active_levels.items()
                        if k in level_filter}
    level_names = list(active_levels.keys())

    ho_base = Path(heldout_dir)
    ho_action_dir = str(ho_base / "embeddings" / "action")
    ho_response_dir = str(ho_base / "embeddings" / "response")
    ho_task_meta = _discover_tasks(heldout_dir, ho_action_dir)

    all_results: dict[str, dict] = {}
    rng = np.random.default_rng(42)

    for train_task in train_tasks:
        tmeta = next(
            (t for t in train_task_meta if t["short_id"] == train_task.label),
            None,
        )
        if tmeta is None:
            continue

        ho_tmeta = next(
            (t for t in ho_task_meta if t["task_id"] == train_task.task_id),
            None,
        )
        if ho_tmeta is None:
            print(f"\n  Skipping {train_task.label} — no heldout data")
            continue

        label = train_task.label
        domain = train_task.domain
        safe_label = label.replace("/", "_").replace("|", "_").replace(":", "_").replace(" ", "_")

        dim = train_task.step_embeddings.shape[1] // 2
        orig_act = train_task.step_embeddings[:, :dim]
        orig_res = train_task.step_embeddings[:, dim:]
        orig_indices = list(train_task.indices)
        orig_rewards = dict(train_task.rewards)
        orig_rids = set(train_task.all_rids)

        ho_rewards = _load_rewards_from_files(ho_tmeta["tdp_files"])
        ho_action_npz = ho_tmeta["npz_path"]
        ho_response_npz = str(Path(ho_response_dir) / Path(ho_action_npz).name)
        if not Path(ho_response_npz).exists():
            continue
        ho_act, ho_res, ho_indices_raw = load_paired_embeddings(
            ho_action_npz, ho_response_npz)

        ho_rid_offset = 20000
        ho_indices = [(r + ho_rid_offset, s) for r, s in ho_indices_raw]
        ho_rewards_shifted = {r + ho_rid_offset: v for r, v in ho_rewards.items()}
        ho_rids = sorted(set(r for r, _ in ho_indices) & set(ho_rewards_shifted.keys()))

        all_act = np.vstack([orig_act, ho_act])
        all_res = np.vstack([orig_res, ho_res])
        all_indices = orig_indices + ho_indices
        all_rewards = {**orig_rewards, **ho_rewards_shifted}
        all_rids = sorted(set(r for r, _ in all_indices) & set(all_rewards.keys()))

        strat_fork_data: dict[str, tuple] = {}
        for strat in FORK_STRATEGIES_ALL:
            if strat not in completed:
                continue
            strat_dir = fork_out / strat / domain
            if not strat_dir.exists():
                continue
            fork_act, fork_res, fork_idx, fork_rew, fork_ti, fork_pmap = \
                _load_fork_data_for_strategy(
                    fork_out, strat, completed[strat], domain, label,
                    safe_label=safe_label, tdp_files=tmeta["tdp_files"],
                )
            if len(fork_rew) == 0:
                continue
            strat_fork_data[strat] = (fork_act, fork_res, fork_idx, fork_rew, fork_ti, fork_pmap)

        conditions = ["baseline"] + [s for s in FORK_STRATEGIES_ALL if s in strat_fork_data]

        n_total = len(all_rids)
        sr = sum(1 for r in all_rids if all_rewards.get(r, 0) >= 0.5) / max(n_total, 1)

        print(f"\n{'#' * 60}")
        print(f"# Fork combined LOO ({cluster_mode}): {label}")
        print(f"#   Original: {len(orig_rids)}, Heldout: {len(ho_rids)}, Total: {n_total}")
        print(f"#   SR={sr:.0%}, Conditions: {conditions}")
        print(f"{'#' * 60}")

        cond_scores: dict[str, dict[str, dict[int, float]]] = {
            lname: {c: {} for c in conditions} for lname in level_names
        }

        for held_out_rid in tqdm(all_rids, desc=f"LOO {_short_task_label(label)}", leave=True):
            train_rids = [r for r in all_rids if r != held_out_rid]
            train_set = set(train_rids)

            train_mask = np.array(
                [i for i, (r, _) in enumerate(all_indices) if r in train_set])
            ho_mask = np.array(
                [i for i, (r, _) in enumerate(all_indices) if r == held_out_rid])

            if len(ho_mask) == 0:
                continue

            ho_act_i = all_act[ho_mask]
            ho_res_i = all_res[ho_mask]
            ho_idx_i = [all_indices[i] for i in ho_mask]

            for cond in conditions:
                fork_pmap: dict[int, tuple[int, int]] = {}

                if cond == "baseline":
                    t_act = all_act[train_mask]
                    t_res = all_res[train_mask]
                    t_indices = [all_indices[i] for i in train_mask]
                    t_rewards = {r: all_rewards[r] for r in train_rids}
                else:
                    fork_act, fork_res, fork_idx, fork_rew, fork_ti, fork_pmap = \
                        strat_fork_data[cond]
                    if recluster:
                        t_act = np.vstack([all_act[train_mask], fork_act])
                        t_res = np.vstack([all_res[train_mask], fork_res])
                        t_indices = [all_indices[i] for i in train_mask] + list(fork_idx)
                    else:
                        t_act = all_act[train_mask]
                        t_res = all_res[train_mask]
                        t_indices = [all_indices[i] for i in train_mask]
                    t_rewards = {r: all_rewards[r] for r in train_rids}
                    t_rewards.update(fork_rew)

                S_train, _, _ = paired_similarity(t_act, t_res, "mean")
                effective_mcs = max(2, min(min_cluster_size, len(t_act) // 5))
                train_labels = hdbscan_cluster_affinity(
                    S_train, min_cluster_size=effective_mcs,
                    noise_strategy=noise_strategy,
                )
                K = int(train_labels.max()) + 1

                _, t_paths = build_global_thought_graph(
                    train_labels, t_indices, t_rewards)
                t_paths = {r: t_paths[r] for r in t_paths if r in t_rewards}
                t_rew = {r: t_rewards[r] for r in t_paths}

                if cond != "baseline" and not recluster and fork_pmap:
                    f_act, f_res = strat_fork_data[cond][:2]
                    f_idx = strat_fork_data[cond][2]
                    f_labels = _assign_heldout_to_clusters(
                        t_act, t_res, train_labels, f_act, f_res)
                    _, f_paths = build_global_thought_graph(
                        f_labels, f_idx, t_rewards)
                    t_paths.update(f_paths)
                    t_rew = {r: t_rewards[r] for r in t_paths}

                sample_weights: dict[int, float] | None = None
                if cond != "baseline" and fork_pmap:
                    parent_child_count: dict[int, int] = {}
                    n_usable = 0
                    for frid, (prid, _fs) in fork_pmap.items():
                        if frid in t_paths and prid in t_paths:
                            parent_child_count[prid] = parent_child_count.get(prid, 0) + 1
                            n_usable += 1
                    drop_frids = []
                    for frid, (prid, fstep) in fork_pmap.items():
                        if frid not in t_paths:
                            continue
                        if prid not in t_paths:
                            drop_frids.append(frid)
                            continue
                        pp = t_paths[prid]
                        t_paths[frid] = pp[:min(fstep + 1, len(pp))] + t_paths[frid]
                    for frid in drop_frids:
                        t_paths.pop(frid, None)
                        t_rew.pop(frid, None)
                    n_real = len(train_set & set(t_paths.keys()))
                    n_tot = n_real + n_usable
                    sample_weights = {}
                    for rid in t_paths:
                        if rid in train_set:
                            sample_weights[rid] = 1.0
                        elif rid in fork_pmap:
                            prid = fork_pmap[rid][0]
                            n_sibs = parent_child_count.get(prid, 1)
                            sample_weights[rid] = (n_real / max(n_tot, 1)) / n_sibs
                        else:
                            sample_weights[rid] = 1.0

                ho_labels_i = _assign_heldout_to_clusters(
                    t_act, t_res, train_labels, ho_act_i, ho_res_i)
                _, ho_paths_i = build_global_thought_graph(
                    ho_labels_i, ho_idx_i, all_rewards)

                if held_out_rid not in ho_paths_i:
                    continue

                for lname, (prm_class, prm_kwargs) in active_levels.items():
                    prm = prm_class(**prm_kwargs)
                    prm.fit(t_paths, t_rew, K, sample_weights=sample_weights)
                    step_sc = prm.score_path(
                        ho_paths_i[held_out_rid], rid=held_out_rid)
                    cond_scores[lname][cond][held_out_rid] = \
                        _aggregate_scores(step_sc, "final")

        outcomes = {r: float(all_rewards.get(r, 0) >= 0.5) for r in all_rids}
        task_result: dict = {
            "task": label, "mode": "fork_combined_loo",
            "cluster_mode": cluster_mode,
            "n_original": len(orig_rids), "n_heldout": len(ho_rids),
            "n_total": n_total, "sr": sr,
            "levels": {},
        }

        for lname in level_names:
            level_d: dict = {"conditions": {}}

            print(f"\n  {lname}:")
            print(f"    {'Condition':<25s} {'AUROC':>7s}  {'BoN@8':>7s}")
            print(f"    {'-'*45}")

            for cond in conditions:
                sc = cond_scores[lname][cond]
                scored_rids = sorted(sc.keys())
                if len(scored_rids) < 3:
                    continue
                auroc = _evaluate_auroc(scored_rids, outcomes, sc)
                bon = _evaluate_bon(scored_rids, outcomes, sc, N_VALUES, rng)
                cond_d: dict = {
                    "auroc": auroc, "n_scored": len(scored_rids),
                    "bon": {N: bon[N] for N in N_VALUES},
                }
                level_d["conditions"][cond] = cond_d
                print(f"    {cond:<25s} {auroc:7.3f}  {bon.get(8, 0):7.3f}")

            task_result["levels"][lname] = level_d

        all_results[label] = task_result

    out_path = results_dir / "fork_combined_loo_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return all_results


def _short_task_label(label: str) -> str:
    if "airplane" in label or "airline" in label:
        return "telecom/Easy"
    if "bad_network" in label:
        return "telecom/Hard"
    if "data_mode" in label:
        return "telecom/None"
    return label


def plot_fork_prm_comparison_multi(
    all_task_results: dict[str, dict],
    out_dir: Path | str,
):
    """Per-task grid: rows = tasks, columns = PRM levels.
    Each cell has grouped bars (baseline + strategies) for AUROC.
    Separate figure for BoN@8.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_labels = sorted(all_task_results.keys())
    if not task_labels:
        return

    sample = all_task_results[task_labels[0]]
    conditions = list(sample["conditions"].keys())
    levels = list(next(iter(sample["conditions"].values()))["levels"].keys())
    n_tasks = len(task_labels)
    n_levels = len(levels)

    level_labels = {
        "level0_markov": "L0: Markov",
        "level1_likelihood": "L1: Likelihood",
        "level2_logreg": "L2: LogReg",
        "level3_fasttext": "L3: FastText",
        "causal_emb_sim": "Emb Cosine",
        "causal_knn": "Emb k-NN",
    }
    cond_labels = {
        "baseline": "Baseline",
        "minority_weighted": "Min. Wt.",
        "minority_early": "Min. Early",
        "early_rare": "Early Rare",
    }
    cond_colors = {
        "baseline": "#4878CF",
        "minority_weighted": "#6ACC65",
        "minority_early": "#D65F5F",
        "early_rare": "#B47CC7",
    }

    x = np.arange(len(conditions))
    bar_width = 0.6

    for metric_name, ylabel, fname in [
        ("auroc", "AUROC", "fork_prm_auroc.png"),
        ("bon8", "BoN@8", "fork_prm_bon8.png"),
    ]:
        fig, axes = plt.subplots(
            n_tasks, n_levels,
            figsize=(3.5 * n_levels, 3.5 * n_tasks),
            squeeze=False,
        )

        for row, tl in enumerate(task_labels):
            tr = all_task_results[tl]
            sr = tr["sr"]
            short = _short_task_label(tl)

            for col, lname in enumerate(levels):
                ax = axes[row, col]
                ll = level_labels.get(lname, lname)

                means = []
                stds = []
                colors = []
                for cond in conditions:
                    sd = tr["conditions"][cond]["levels"][lname]["strategies"]["final"]
                    if metric_name == "auroc":
                        means.append(sd["auroc_mean"])
                        stds.append(sd["auroc_std"])
                    else:
                        means.append(sd["bon"]["8"]["mean"])
                        stds.append(sd["bon"]["8"]["std"])
                    colors.append(cond_colors.get(cond, "#999999"))

                means_arr = np.array(means)
                stds_arr = np.array(stds)
                lo = np.clip(means_arr - stds_arr, 0, 1)
                hi = np.clip(means_arr + stds_arr, 0, 1)

                ax.bar(
                    x, means_arr, bar_width,
                    yerr=[means_arr - lo, hi - means_arr],
                    capsize=3, color=colors, alpha=0.85,
                    edgecolor="black", linewidth=0.4,
                )

                if metric_name == "auroc":
                    ax.axhline(0.5, color="gray", ls="--", lw=0.7, alpha=0.5)
                    ax.set_ylim(0.3, 1.05)
                else:
                    ax.axhline(sr, color="gray", ls="--", lw=0.7, alpha=0.5)
                    ax.set_ylim(max(0, sr - 0.2), 1.05)

                for i, v in enumerate(means):
                    ax.text(i, min(hi[i] + 0.01, 1.04), f"{v:.2f}",
                            ha="center", va="bottom", fontsize=6.5)

                ax.set_xticks(x)
                if row == n_tasks - 1:
                    ax.set_xticklabels(
                        [cond_labels.get(c, c) for c in conditions],
                        fontsize=7, rotation=30, ha="right",
                    )
                else:
                    ax.set_xticklabels([])

                if row == 0:
                    ax.set_title(ll, fontsize=10, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(f"{short}\n(sr={sr:.0%})\n{ylabel}",
                                  fontsize=9)

                ax.grid(True, alpha=0.15, axis="y")

        fig.suptitle(
            f"Fork-Augmented PRM — {ylabel} "
            f"(80/20 split, {sample['n_repeats']} repeats)",
            fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        png_path = out_dir / fname
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {png_path}")

    # Also generate the combined single-figure comparison
    _plot_fork_prm_combined(all_task_results, out_dir)


def _plot_fork_prm_combined(
    all_task_results: dict[str, dict],
    out_dir: Path,
):
    """Single comparison figure: one row per metric (AUROC, BoN@8),
    columns = PRM levels. Grouped bars = conditions, hatch/pattern = tasks.
    """
    task_labels = sorted(all_task_results.keys())
    sample = all_task_results[task_labels[0]]
    conditions = list(sample["conditions"].keys())
    levels = list(next(iter(sample["conditions"].values()))["levels"].keys())

    level_labels = {
        "level0_markov": "L0: Markov",
        "level1_likelihood": "L1: Likelihood",
        "level2_logreg": "L2: LogReg",
        "level3_fasttext": "L3: FastText",
        "causal_emb_sim": "Emb Cosine",
        "causal_knn": "Emb k-NN",
    }
    cond_labels = {
        "baseline": "Baseline",
        "minority_weighted": "Min. Weighted",
        "minority_early": "Min. Early",
        "early_rare": "Early Rare",
    }

    n_tasks = len(task_labels)
    n_conds = len(conditions)
    n_levels = len(levels)

    group_width = 0.8
    bar_w = group_width / n_conds
    task_colors = ["#4878CF", "#D65F5F", "#6ACC65"][:n_tasks]

    for metric_name, ylabel, fname in [
        ("auroc", "AUROC", "fork_prm_comparison.png"),
    ]:
        fig, axes = plt.subplots(
            n_tasks, n_levels,
            figsize=(4 * n_levels, 3.5 * n_tasks),
            squeeze=False,
        )

        for row, tl in enumerate(task_labels):
            tr = all_task_results[tl]
            sr = tr["sr"]
            short = _short_task_label(tl)

            for col, lname in enumerate(levels):
                ax = axes[row, col]
                ll = level_labels.get(lname, lname)
                x = np.arange(n_conds)

                means = []
                stds = []
                for cond in conditions:
                    sd = tr["conditions"][cond]["levels"][lname]["strategies"]["final"]
                    means.append(sd["auroc_mean"])
                    stds.append(sd["auroc_std"])

                means_arr = np.array(means)
                stds_arr = np.array(stds)
                lo = np.clip(means_arr - stds_arr, 0, 1)
                hi = np.clip(means_arr + stds_arr, 0, 1)

                ax.bar(
                    x, means_arr, 0.6,
                    yerr=[means_arr - lo, hi - means_arr],
                    capsize=3, color=task_colors[row], alpha=0.85,
                    edgecolor="black", linewidth=0.4,
                )
                ax.axhline(0.5, color="gray", ls="--", lw=0.7, alpha=0.5)
                ax.set_ylim(0.3, 1.05)

                for i, v in enumerate(means):
                    ax.text(i, min(hi[i] + 0.01, 1.04), f"{v:.3f}",
                            ha="center", va="bottom", fontsize=7)

                ax.set_xticks(x)
                if row == n_tasks - 1:
                    ax.set_xticklabels(
                        [cond_labels.get(c, c) for c in conditions],
                        fontsize=7, rotation=30, ha="right",
                    )
                else:
                    ax.set_xticklabels([])

                if row == 0:
                    ax.set_title(ll, fontsize=10, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(f"{short} (sr={sr:.0%})\nAUROC", fontsize=9)
                ax.grid(True, alpha=0.15, axis="y")

        fig.suptitle(
            f"Fork-Augmented PRM — AUROC by Task & Strategy "
            f"({sample['n_repeats']} repeats, 80/20 split)",
            fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_dir / fname}")


# ── Held-out & combined-LOO evaluation ────────────────────────────────


def _load_heldout_for_task(
    train_task: TaskData,
    heldout_dir: str,
) -> dict | None:
    """Load heldout rollouts for a matching task and NN-assign to train clusters.

    Returns dict with keys: rewards, rollout_paths, all_rids, outcome, K,
    or None if no matching heldout data is found.
    """
    ho_base = Path(heldout_dir)
    ho_action_dir = str(ho_base / "embeddings" / "action")
    ho_response_dir = str(ho_base / "embeddings" / "response")

    ho_tasks = _discover_tasks(heldout_dir, ho_action_dir)
    match = None
    for ht in ho_tasks:
        if ht["task_id"] == train_task.task_id:
            match = ht
            break
    if match is None:
        return None

    ho_rewards = _load_rewards_from_files(match["tdp_files"])
    ho_action_npz = match["npz_path"]
    ho_response_npz = str(Path(ho_response_dir) / Path(ho_action_npz).name)
    if not Path(ho_response_npz).exists():
        return None

    ho_act, ho_res, ho_indices = load_paired_embeddings(ho_action_npz, ho_response_npz)

    # NN-assign heldout steps to training clusters
    train_act_npz = None
    train_action_dir = str(Path(train_task.domain).parent)  # unused, use task info
    # Reconstruct train embeddings from task data
    # train_task.step_embeddings is hstack(act, res), indices is (rid, step)
    dim = train_task.step_embeddings.shape[1] // 2
    train_act = train_task.step_embeddings[:, :dim]
    train_res = train_task.step_embeddings[:, dim:]

    ho_labels = _assign_heldout_to_clusters(
        train_act, train_res, train_task.labels,
        ho_act, ho_res,
    )

    _, ho_rollout_paths = build_global_thought_graph(
        ho_labels, ho_indices, ho_rewards,
    )
    ho_rids = sorted(set(ho_rollout_paths.keys()) & set(ho_rewards.keys()))
    ho_rollout_paths = {r: ho_rollout_paths[r] for r in ho_rids}
    ho_outcome = np.array(
        [1.0 if ho_rewards.get(r, 0) >= 0.5 else 0.0 for r in ho_rids]
    )

    return {
        "rewards": ho_rewards,
        "rollout_paths": ho_rollout_paths,
        "all_rids": ho_rids,
        "outcome": ho_outcome,
        "K": train_task.K,
    }


def run_heldout_eval(
    data_dir: str,
    heldout_dir: str,
    out_dir: str,
    min_cluster_size: int = 5,
    noise_strategy: str = "outlier_bin",
    tool_arg_beta: float = 0.0,
    level_filter: set[str] | None = None,
) -> dict:
    """Option A: Train PRMs on original data, evaluate on held-out rollouts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HELD-OUT EVALUATION")
    print(f"  Train data:  {data_dir}")
    print(f"  Heldout data: {heldout_dir}")
    print(f"  noise_strategy = {noise_strategy!r}")
    print("=" * 70)

    train_tasks = load_all_tasks(
        data_dir, min_cluster_size=min_cluster_size,
        noise_strategy=noise_strategy, tool_arg_beta=tool_arg_beta,
    )

    all_results: dict[str, dict] = {}
    rng = np.random.default_rng(42)

    for train_task in train_tasks:
        ho = _load_heldout_for_task(train_task, heldout_dir)
        if ho is None:
            print(f"\n  Skipping {train_task.label} — no heldout data")
            continue

        n_ho = len(ho["all_rids"])
        n_ho_pass = int(ho["outcome"].sum())
        sr_ho = n_ho_pass / n_ho if n_ho else 0
        sr_train = float(np.mean(train_task.outcome))

        print(f"\n{'#' * 70}")
        print(f"# {train_task.label}")
        print(f"#   Train: {len(train_task.all_rids)} rollouts, SR={sr_train:.0%}")
        print(f"#   Test:  {n_ho} rollouts, SR={sr_ho:.0%} "
              f"({n_ho_pass} pass / {n_ho - n_ho_pass} fail)")
        print(f"{'#' * 70}")

        task_dir = out / train_task.label.replace("/", "_")
        task_dir.mkdir(parents=True, exist_ok=True)
        task_results = {}

        task_levels = _build_levels_for_task(train_task, level_filter=level_filter)
        outcomes_ho = {r: float(ho["rewards"].get(r, 0) >= 0.5)
                       for r in ho["all_rids"]}

        for level_name, (prm_class, prm_kwargs) in task_levels.items():
            print(f"\n  {level_name}...")

            prm = prm_class(**prm_kwargs)
            prm.fit(train_task.rollout_paths, train_task.rewards, train_task.K)

            step_scores: dict[int, list[float]] = {}
            for r in ho["all_rids"]:
                step_scores[r] = prm.score_path(ho["rollout_paths"][r], rid=r)

            result: dict = {
                "task": train_task.label,
                "mode": "heldout",
                "n_train": len(train_task.all_rids),
                "n_test": n_ho,
                "sr_train": sr_train,
                "sr_test": sr_ho,
                "K": train_task.K,
                "strategies": {},
            }

            for strat in STRATEGIES:
                agg = {r: _aggregate_scores(step_scores[r], strat)
                       for r in ho["all_rids"]}
                auroc = _evaluate_auroc(ho["all_rids"], outcomes_ho, agg)
                bon = _evaluate_bon(
                    ho["all_rids"], outcomes_ho, agg, N_VALUES, rng,
                )
                strat_d: dict = {
                    "auroc": auroc,
                    "bon": {},
                }
                for N in N_VALUES:
                    strat_d["bon"][N] = {
                        "value": bon[N],
                        "random": sr_ho,
                        "oracle": 1 - (1 - sr_ho) ** N,
                    }
                result["strategies"][strat] = strat_d

            auroc_final = result["strategies"]["final"]["auroc"]
            bon8 = result["strategies"]["final"]["bon"][8]["value"]
            print(f"    final: AUROC={auroc_final:.3f}  BoN@8={bon8:.3f}")

            with open(task_dir / f"{level_name}.json", "w") as f:
                json.dump(result, f, indent=2)
            task_results[level_name] = result

        all_results[train_task.label] = task_results

    # Summary
    print(f"\n{'=' * 70}")
    print("HELD-OUT EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    header = (f"{'Task':<30s} {'Method':<20s} {'TrainSR':>7s} "
              f"{'TestSR':>7s} {'AUROC':>7s} {'BoN@8':>7s}")
    print(f"\n{header}")
    print("-" * len(header))
    for tl in sorted(all_results.keys()):
        for lname in ALL_METHODS:
            if lname not in all_results[tl]:
                continue
            r = all_results[tl][lname]
            auroc = r["strategies"]["final"]["auroc"]
            bon8 = r["strategies"]["final"]["bon"][8]["value"]
            print(
                f"{tl:<30s} {LEVEL_LABELS.get(lname, lname):<20s} "
                f"{r['sr_train']:>7.0%} {r['sr_test']:>7.0%} "
                f"{auroc:>7.3f} {bon8:>7.3f}"
            )

    with open(out / "heldout_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out / 'heldout_results.json'}")
    return all_results


def run_augmented_heldout_eval(
    data_dir: str,
    extra_train_dir: str,
    heldout_dir: str,
    out_dir: str,
    min_cluster_size: int = 5,
    noise_strategy: str = "outlier_bin",
    tool_arg_beta: float = 0.0,
    level_filter: set[str] | None = None,
) -> dict:
    """Train PRMs on original + extra training data, evaluate on held-out.

    Merges embeddings from *data_dir* (e.g. original 100) and
    *extra_train_dir* (e.g. new 290), reclusters on the combined set,
    then evaluates on *heldout_dir* (e.g. original 200).
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AUGMENTED HELD-OUT EVALUATION")
    print(f"  Original train: {data_dir}")
    print(f"  Extra train:    {extra_train_dir}")
    print(f"  Heldout data:   {heldout_dir}")
    print(f"  MCS={min_cluster_size}  noise_strategy={noise_strategy!r}")
    print("=" * 70)

    # --- Load original training data (pre-clustered) ---
    train_tasks = load_all_tasks(
        data_dir, min_cluster_size=min_cluster_size,
        noise_strategy=noise_strategy, tool_arg_beta=tool_arg_beta,
    )

    base_orig = Path(data_dir)
    orig_task_meta = _discover_tasks(data_dir, str(base_orig / "embeddings" / "action"))

    # --- Discover extra training data ---
    base_extra = Path(extra_train_dir)
    extra_action_dir = str(base_extra / "embeddings" / "action")
    extra_response_dir = str(base_extra / "embeddings" / "response")
    extra_task_meta = _discover_tasks(extra_train_dir, extra_action_dir)

    # --- Discover heldout data ---
    ho_base = Path(heldout_dir)
    ho_action_dir = str(ho_base / "embeddings" / "action")
    ho_response_dir = str(ho_base / "embeddings" / "response")
    ho_task_meta = _discover_tasks(heldout_dir, ho_action_dir)

    active_levels = dict(LEVELS)
    if level_filter:
        active_levels = {k: v for k, v in active_levels.items()
                        if k in level_filter}

    all_results: dict[str, dict] = {}
    rng = np.random.default_rng(42)

    for train_task in train_tasks:
        tmeta = next(
            (t for t in orig_task_meta if t["short_id"] == train_task.label),
            None,
        )
        if tmeta is None:
            continue

        # Match extra train task by task_id
        extra_tmeta = next(
            (t for t in extra_task_meta if t["task_id"] == train_task.task_id),
            None,
        )
        if extra_tmeta is None:
            print(f"\n  Skipping {train_task.label} — no extra train data")
            continue

        # Match heldout task
        ho_tmeta = next(
            (t for t in ho_task_meta if t["task_id"] == train_task.task_id),
            None,
        )
        if ho_tmeta is None:
            print(f"\n  Skipping {train_task.label} — no heldout data")
            continue

        label = train_task.label

        # --- Load original embeddings ---
        dim = train_task.step_embeddings.shape[1] // 2
        orig_act = train_task.step_embeddings[:, :dim]
        orig_res = train_task.step_embeddings[:, dim:]
        orig_indices = list(train_task.indices)
        orig_rewards = dict(train_task.rewards)
        orig_rids = set(train_task.all_rids)

        # --- Load extra training embeddings ---
        extra_action_npz = extra_tmeta["npz_path"]
        extra_response_npz = str(Path(extra_response_dir) / Path(extra_action_npz).name)
        if not Path(extra_response_npz).exists():
            print(f"\n  Skipping {label} — missing extra response NPZ")
            continue
        extra_act, extra_res, extra_indices_raw = load_paired_embeddings(
            extra_action_npz, extra_response_npz)
        extra_rewards = _load_rewards_from_files(extra_tmeta["tdp_files"])

        extra_rid_offset = 30000
        extra_indices = [(r + extra_rid_offset, s) for r, s in extra_indices_raw]
        extra_rewards_shifted = {r + extra_rid_offset: v for r, v in extra_rewards.items()}

        # --- Merge training data ---
        all_train_act = np.vstack([orig_act, extra_act])
        all_train_res = np.vstack([orig_res, extra_res])
        all_train_indices = orig_indices + extra_indices
        all_train_rewards = {**orig_rewards, **extra_rewards_shifted}
        all_train_rids = sorted(
            set(r for r, _ in all_train_indices) & set(all_train_rewards.keys()))

        n_orig = len(orig_rids)
        n_extra = len(set(r for r, _ in extra_indices) & set(extra_rewards_shifted.keys()))

        # --- Cluster on combined training data ---
        S_train, _, _ = paired_similarity(all_train_act, all_train_res, "mean")
        if tool_arg_beta > 0:
            orig_tool_info = _extract_step_tool_info(tmeta["tdp_files"], orig_indices)
            extra_tool_info = _extract_step_tool_info(extra_tmeta["tdp_files"],
                                                       extra_indices_raw)
            combined_tool_info = orig_tool_info + extra_tool_info
            S_train = apply_tool_arg_penalty(S_train, combined_tool_info,
                                             beta=tool_arg_beta)
        all_train_labels = hdbscan_cluster_affinity(
            S_train, min_cluster_size=min_cluster_size,
            noise_strategy=noise_strategy,
        )
        K = int(all_train_labels.max()) + 1

        _, train_rollout_paths = build_global_thought_graph(
            all_train_labels, all_train_indices, all_train_rewards,
        )
        train_rew_filtered = {r: all_train_rewards[r]
                              for r in all_train_rids if r in train_rollout_paths}
        sr_train = float(np.mean([1.0 if v >= 0.5 else 0.0
                                  for v in train_rew_filtered.values()]))

        # --- Load held-out data ---
        ho_action_npz = ho_tmeta["npz_path"]
        ho_response_npz = str(Path(ho_response_dir) / Path(ho_action_npz).name)
        if not Path(ho_response_npz).exists():
            continue
        ho_act, ho_res, ho_indices = load_paired_embeddings(
            ho_action_npz, ho_response_npz)
        ho_rewards = _load_rewards_from_files(ho_tmeta["tdp_files"])

        # --- NN-assign held-out to combined training clusters ---
        ho_labels = _assign_heldout_to_clusters(
            all_train_act, all_train_res, all_train_labels,
            ho_act, ho_res,
        )
        _, ho_rollout_paths = build_global_thought_graph(
            ho_labels, ho_indices, ho_rewards,
        )
        ho_rids = sorted(set(ho_rollout_paths.keys()) & set(ho_rewards.keys()))
        n_ho = len(ho_rids)
        ho_outcome = np.array(
            [1.0 if ho_rewards.get(r, 0) >= 0.5 else 0.0 for r in ho_rids])
        sr_ho = float(ho_outcome.mean()) if n_ho else 0
        outcomes_ho = {r: float(ho_rewards.get(r, 0) >= 0.5) for r in ho_rids}

        print(f"\n{'#' * 70}")
        print(f"# {label}")
        print(f"#   Train: {n_orig} orig + {n_extra} extra = "
              f"{len(all_train_rids)} rollouts, SR={sr_train:.0%}, K={K}")
        print(f"#   Test:  {n_ho} rollouts, SR={sr_ho:.0%}")
        print(f"{'#' * 70}")

        # --- Build embedding baselines from combined training data ---
        all_train_step_emb = np.hstack([all_train_act, all_train_res])
        rid_to_rows: dict[int, list[int]] = {}
        for i, (rid, _step) in enumerate(all_train_indices):
            rid_to_rows.setdefault(rid, []).append(i)

        task_levels = dict(active_levels)
        emb_args = {
            "step_embeddings": all_train_step_emb,
            "rid_to_rows": rid_to_rows,
        }
        task_levels["causal_emb_sim"] = (CausalEmbSimBaseline, dict(emb_args))
        task_levels["causal_knn"] = (CausalKnnBaseline, {**emb_args, "k": 5})
        if level_filter:
            task_levels = {k: v for k, v in task_levels.items()
                          if k in level_filter}

        task_dir = out / label.replace("/", "_")
        task_dir.mkdir(parents=True, exist_ok=True)
        task_results = {}

        for level_name, (prm_class, prm_kwargs) in task_levels.items():
            print(f"\n  {level_name}...")

            prm = prm_class(**prm_kwargs)
            prm.fit(train_rollout_paths, train_rew_filtered, K)

            step_scores: dict[int, list[float]] = {}
            for r in ho_rids:
                if r in ho_rollout_paths:
                    step_scores[r] = prm.score_path(ho_rollout_paths[r], rid=r)

            scored_rids = [r for r in ho_rids if r in step_scores]

            result: dict = {
                "task": label,
                "mode": "augmented_heldout",
                "n_orig": n_orig,
                "n_extra": n_extra,
                "n_train": len(all_train_rids),
                "n_test": n_ho,
                "sr_train": sr_train,
                "sr_test": sr_ho,
                "K": K,
                "strategies": {},
            }

            for strat in STRATEGIES:
                agg = {r: _aggregate_scores(step_scores[r], strat)
                       for r in scored_rids}
                auroc = _evaluate_auroc(scored_rids, outcomes_ho, agg)
                bon = _evaluate_bon(scored_rids, outcomes_ho, agg, N_VALUES, rng)
                strat_d: dict = {"auroc": auroc, "bon": {}}
                for N in N_VALUES:
                    strat_d["bon"][N] = {
                        "value": bon[N],
                        "random": sr_ho,
                        "oracle": 1 - (1 - sr_ho) ** N,
                    }
                result["strategies"][strat] = strat_d

            auroc_final = result["strategies"]["final"]["auroc"]
            bon8 = result["strategies"]["final"]["bon"][8]["value"]
            print(f"    final: AUROC={auroc_final:.3f}  BoN@8={bon8:.3f}")

            with open(task_dir / f"{level_name}.json", "w") as f:
                json.dump(result, f, indent=2)
            task_results[level_name] = result

        all_results[label] = task_results

    # Summary
    print(f"\n{'=' * 70}")
    print("AUGMENTED HELD-OUT EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    header = (f"{'Task':<30s} {'Method':<20s} {'TrainSR':>7s} "
              f"{'TestSR':>7s} {'AUROC':>7s} {'BoN@8':>7s}")
    print(f"\n{header}")
    print("-" * len(header))
    for tl in sorted(all_results.keys()):
        for lname in ALL_METHODS:
            if lname not in all_results[tl]:
                continue
            r = all_results[tl][lname]
            auroc = r["strategies"]["final"]["auroc"]
            bon8 = r["strategies"]["final"]["bon"][8]["value"]
            print(
                f"{tl:<30s} {LEVEL_LABELS.get(lname, lname):<20s} "
                f"{r['sr_train']:>7.0%} {r['sr_test']:>7.0%} "
                f"{auroc:>7.3f} {bon8:>7.3f}"
            )

    with open(out / "augmented_heldout_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out / 'augmented_heldout_results.json'}")
    return all_results


def _load_combined_task(
    train_task: TaskData,
    heldout_dir: str,
    min_cluster_size: int = 5,
    noise_strategy: str = "outlier_bin",
    tool_arg_beta: float = 0.0,
    recluster: bool = False,
    extra_dir: str | None = None,
) -> dict | None:
    """Load and merge original + heldout (+ optional extra) into combined data.

    If recluster=True, re-runs HDBSCAN on the concatenated embeddings.
    If recluster=False, NN-assigns heldout/extra to original clusters.

    When *extra_dir* is provided, its rollouts are included with a separate
    RID offset (40000) so they don't collide with originals or heldout.
    """
    ho_base = Path(heldout_dir)
    ho_action_dir = str(ho_base / "embeddings" / "action")
    ho_response_dir = str(ho_base / "embeddings" / "response")

    ho_tasks = _discover_tasks(heldout_dir, ho_action_dir)
    match = None
    for ht in ho_tasks:
        if ht["task_id"] == train_task.task_id:
            match = ht
            break
    if match is None:
        return None

    ho_rewards = _load_rewards_from_files(match["tdp_files"])
    ho_action_npz = match["npz_path"]
    ho_response_npz = str(Path(ho_response_dir) / Path(ho_action_npz).name)
    if not Path(ho_response_npz).exists():
        return None

    ho_act, ho_res, ho_indices = load_paired_embeddings(ho_action_npz, ho_response_npz)

    dim = train_task.step_embeddings.shape[1] // 2
    train_act = train_task.step_embeddings[:, :dim]
    train_res = train_task.step_embeddings[:, dim:]

    max_train_rid = max(train_task.all_rids) if train_task.all_rids else 0
    ho_rid_offset = max_train_rid + 1

    # --- Optionally load extra training data ---
    extra_act = extra_res = extra_indices_raw = None
    extra_rewards: dict[int, float] = {}
    extra_rid_offset = 40000
    n_extra = 0
    if extra_dir:
        ex_base = Path(extra_dir)
        ex_action_dir = str(ex_base / "embeddings" / "action")
        ex_response_dir = str(ex_base / "embeddings" / "response")
        ex_tasks = _discover_tasks(extra_dir, ex_action_dir)
        ex_match = next(
            (t for t in ex_tasks if t["task_id"] == train_task.task_id), None)
        if ex_match is not None:
            ex_action_npz = ex_match["npz_path"]
            ex_response_npz = str(Path(ex_response_dir) / Path(ex_action_npz).name)
            if Path(ex_response_npz).exists():
                extra_act, extra_res, extra_indices_raw = load_paired_embeddings(
                    ex_action_npz, ex_response_npz)
                extra_rewards = _load_rewards_from_files(ex_match["tdp_files"])
                n_extra = len(set(r for r, _ in extra_indices_raw)
                              & set(extra_rewards.keys()))

    if recluster:
        parts_act = [train_act, ho_act]
        parts_res = [train_res, ho_res]
        ho_indices_remapped = [(r + ho_rid_offset, s) for r, s in ho_indices]
        all_indices = list(train_task.indices) + ho_indices_remapped

        combined_rewards = dict(train_task.rewards)
        for r in ho_rewards:
            combined_rewards[r + ho_rid_offset] = ho_rewards[r]

        if extra_act is not None:
            parts_act.append(extra_act)
            parts_res.append(extra_res)
            ex_indices_remapped = [(r + extra_rid_offset, s)
                                   for r, s in extra_indices_raw]
            all_indices = all_indices + ex_indices_remapped
            for r in extra_rewards:
                combined_rewards[r + extra_rid_offset] = extra_rewards[r]

        all_act = np.vstack(parts_act)
        all_res = np.vstack(parts_res)

        S_paired, _, _ = paired_similarity(all_act, all_res, "mean")
        if tool_arg_beta > 0:
            pass  # tool-arg penalty skipped in recluster mode
        all_labels = hdbscan_cluster_affinity(
            S_paired, min_cluster_size=min_cluster_size,
            noise_strategy=noise_strategy,
        )
        K = int(all_labels.max()) + 1

        _, combined_paths = build_global_thought_graph(
            all_labels, all_indices, combined_rewards,
        )
    else:
        ho_labels = _assign_heldout_to_clusters(
            train_act, train_res, train_task.labels,
            ho_act, ho_res,
        )
        _, ho_rollout_paths = build_global_thought_graph(
            ho_labels, ho_indices, ho_rewards,
        )
        K = train_task.K
        combined_paths = dict(train_task.rollout_paths)
        combined_rewards = dict(train_task.rewards)
        for r in sorted(set(ho_rollout_paths.keys()) & set(ho_rewards.keys())):
            combined_paths[r + ho_rid_offset] = ho_rollout_paths[r]
            combined_rewards[r + ho_rid_offset] = ho_rewards[r]

        if extra_act is not None:
            ex_labels = _assign_heldout_to_clusters(
                train_act, train_res, train_task.labels,
                extra_act, extra_res,
            )
            _, ex_rollout_paths = build_global_thought_graph(
                ex_labels, extra_indices_raw, extra_rewards,
            )
            for r in sorted(set(ex_rollout_paths.keys()) & set(extra_rewards.keys())):
                combined_paths[r + extra_rid_offset] = ex_rollout_paths[r]
                combined_rewards[r + extra_rid_offset] = extra_rewards[r]

    combined_rids = sorted(set(combined_paths.keys()) & set(combined_rewards.keys()))
    combined_paths = {r: combined_paths[r] for r in combined_rids}
    combined_outcome = np.array(
        [1.0 if combined_rewards.get(r, 0) >= 0.5 else 0.0
         for r in combined_rids]
    )

    return {
        "paths": combined_paths,
        "rewards": combined_rewards,
        "rids": combined_rids,
        "outcome": combined_outcome,
        "K": K,
        "n_original": len(train_task.all_rids),
        "n_heldout": len(ho_rewards),
        "n_extra": n_extra,
    }


def run_combined_loo(
    data_dir: str,
    heldout_dir: str,
    out_dir: str,
    min_cluster_size: int = 5,
    noise_strategy: str = "outlier_bin",
    tool_arg_beta: float = 0.0,
    level_filter: set[str] | None = None,
    recluster: bool = False,
    extra_train_dir: str | None = None,
) -> dict:
    """Option B: LOO-CV on combined original + heldout (+ optional extra) rollouts.

    If recluster=False (default), clusters on originals and NN-assigns heldout.
    If recluster=True, re-runs HDBSCAN on all combined embeddings.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cluster_mode = "recluster" if recluster else "fixed topology"
    print("=" * 70)
    print(f"COMBINED LOO-CV EVALUATION ({cluster_mode})")
    print(f"  Original data: {data_dir}")
    print(f"  Heldout data:  {heldout_dir}")
    if extra_train_dir:
        print(f"  Extra data:    {extra_train_dir}")
    print(f"  noise_strategy = {noise_strategy!r}")
    print("=" * 70)

    train_tasks = load_all_tasks(
        data_dir, min_cluster_size=min_cluster_size,
        noise_strategy=noise_strategy, tool_arg_beta=tool_arg_beta,
    )

    all_results: dict[str, dict] = {}

    for train_task in train_tasks:
        combined = _load_combined_task(
            train_task, heldout_dir,
            min_cluster_size=min_cluster_size,
            noise_strategy=noise_strategy,
            tool_arg_beta=tool_arg_beta,
            recluster=recluster,
            extra_dir=extra_train_dir,
        )
        if combined is None:
            print(f"\n  Skipping {train_task.label} — no heldout data")
            continue

        combined_paths = combined["paths"]
        combined_rewards = combined["rewards"]
        combined_rids = combined["rids"]
        combined_outcome = combined["outcome"]
        K = combined["K"]
        n_total = len(combined_rids)
        n_pass = int(combined_outcome.sum())
        sr = n_pass / n_total if n_total else 0

        n_extra = combined.get("n_extra", 0)
        print(f"\n{'#' * 70}")
        print(f"# {train_task.label}  (combined, {cluster_mode}, K={K})")
        extra_str = f", Extra: {n_extra}" if n_extra else ""
        print(f"#   Original: {combined['n_original']}, "
              f"Heldout: {combined['n_heldout']}{extra_str}, "
              f"Total: {n_total}, SR={sr:.0%} "
              f"({n_pass} pass / {n_total - n_pass} fail)")
        print(f"{'#' * 70}")

        task_dir = out / train_task.label.replace("/", "_")
        task_dir.mkdir(parents=True, exist_ok=True)
        task_results = {}

        task_levels = _build_levels_for_task(train_task, level_filter=level_filter)
        outcomes = {r: float(combined_rewards.get(r, 0) >= 0.5)
                    for r in combined_rids}

        for level_name, (prm_class, prm_kwargs) in task_levels.items():
            print(f"\n  {level_name} (LOO on {n_total} rollouts)...")

            scores_per_rid: dict[int, float] = {}
            from tqdm import tqdm as _tqdm
            for i, held_out in enumerate(_tqdm(
                combined_rids, desc=f"    {level_name}", leave=False,
            )):
                train_rids_loo = [r for r in combined_rids if r != held_out]
                train_paths_loo = {r: combined_paths[r] for r in train_rids_loo}
                train_rewards_loo = {r: combined_rewards[r] for r in train_rids_loo}

                prm = prm_class(**prm_kwargs)
                prm.fit(train_paths_loo, train_rewards_loo, K)

                step_sc = prm.score_path(combined_paths[held_out], rid=held_out)
                scores_per_rid[held_out] = _aggregate_scores(step_sc, "final")

            auroc = _evaluate_auroc(combined_rids, outcomes, scores_per_rid)
            rng = np.random.default_rng(42)
            bon = _evaluate_bon(combined_rids, outcomes, scores_per_rid,
                                N_VALUES, rng)

            result: dict = {
                "task": train_task.label,
                "mode": "combined_loo",
                "n_original": combined["n_original"],
                "n_heldout": combined["n_heldout"],
                "n_extra": combined.get("n_extra", 0),
                "n_total": n_total,
                "sr": sr,
                "K": K,
                "auroc": auroc,
                "bon": {},
            }
            for N in N_VALUES:
                result["bon"][N] = {
                    "value": bon[N],
                    "random": sr,
                    "oracle": 1 - (1 - sr) ** N,
                }

            print(f"    AUROC={auroc:.3f}  BoN@8={bon.get(8, 0):.3f}")

            with open(task_dir / f"{level_name}.json", "w") as f:
                json.dump(result, f, indent=2)
            task_results[level_name] = result

        all_results[train_task.label] = task_results

    # Summary
    print(f"\n{'=' * 70}")
    print("COMBINED LOO-CV SUMMARY")
    print(f"{'=' * 70}")
    header = (f"{'Task':<30s} {'Method':<20s} {'N':>5s} "
              f"{'SR':>5s} {'AUROC':>7s} {'BoN@8':>7s}")
    print(f"\n{header}")
    print("-" * len(header))
    for tl in sorted(all_results.keys()):
        for lname in ALL_METHODS:
            if lname not in all_results[tl]:
                continue
            r = all_results[tl][lname]
            auroc = r["auroc"]
            bon8 = r["bon"].get("8", r["bon"].get(8, {})).get("value", 0)
            print(
                f"{tl:<30s} {LEVEL_LABELS.get(lname, lname):<20s} "
                f"{r['n_total']:>5d} {r['sr']:>5.0%} "
                f"{auroc:>7.3f} {bon8:>7.3f}"
            )

    with open(out / "combined_loo_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out / 'combined_loo_results.json'}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Cluster-path PRM evaluation")
    parser.add_argument(
        "--data-dir",
        default="exp/tau2/baseline/multi_task_100/data",
        help="Experiment data directory",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--mcs", type=int, default=5, help="HDBSCAN min_cluster_size")
    parser.add_argument(
        "--noise-strategy", default="outlier_bin",
        choices=["nearest", "outlier_bin"],
        help="How to handle HDBSCAN outlier points: 'outlier_bin' collects "
             "outliers into a single extra cluster (default), 'nearest' "
             "reassigns to closest cluster",
    )
    parser.add_argument("--plots-only", action="store_true", help="Replot from saved results")
    parser.add_argument("--complexity", action="store_true", help="Analyze complexity correlation")
    parser.add_argument("--step-bon", action="store_true", help="Compute step-level BoN metric")
    parser.add_argument("--disagree", action="store_true", help="Analyze PRM disagreement points")
    parser.add_argument("--manifest", action="store_true", help="Build branching manifest from high-margin errors")
    parser.add_argument("--n-targets", type=int, default=50, help="Number of branching targets")
    parser.add_argument("--subsample", action="store_true",
                        help="Run rollout subsampling experiment (30-90 rollouts)")
    parser.add_argument("--mcs-sweep", type=int, nargs="+", default=None,
                        help="HDBSCAN min_cluster_size values to sweep in subsample "
                             "experiment (default: just --mcs)")
    parser.add_argument("--convergence", action="store_true",
                        help="Measure cluster partition convergence (ARI/NMI) vs subsample size")
    parser.add_argument("--subsample-hard", action="store_true",
                        help="Hard subsampling: cluster + train PRM on N rollouts, "
                             "evaluate AUROC/BoN on held-out rollouts")
    parser.add_argument("--biased-subsample", action="store_true",
                        help="Run biased branch-point subsampling experiment "
                             "(minority-outcome oversampling) on target tasks")
    parser.add_argument("--diagnose-only", action="store_true",
                        help="With --biased-subsample or --fork: only print the "
                             "diagnostic, don't execute")
    parser.add_argument("--fork", action="store_true",
                        help="Run fork-and-continue minority branch experiment "
                             "on target tasks (uses cold_fork_from_tdp)")
    parser.add_argument("--fork-recluster", action="store_true",
                        help="Re-cluster original + forked rollouts per strategy "
                             "and render comparative flow diagrams")
    parser.add_argument("--fork-prm", action="store_true",
                        help="Train PRMs on fork-augmented data and evaluate "
                             "against baseline on held-out originals")
    parser.add_argument("--fork-loo", action="store_true",
                        help="LOO-CV on original rollouts with fork-augmented "
                             "training sets. Use --recluster for combined "
                             "clustering, default is NN-assign.")
    parser.add_argument("--fork-data-dir", default=None,
                        help="Directory with fork strategy subdirs, embed "
                             "caches, and fork_results.jsonl (default: "
                             "out-dir/minority_fork)")
    parser.add_argument("--fork-heldout", action="store_true",
                        help="Train PRMs on originals + cold forks, evaluate "
                             "on --heldout-dir. Use --recluster for combined "
                             "clustering.")
    parser.add_argument("--fork-combined-loo", action="store_true",
                        help="LOO-CV on combined originals + heldout, with "
                             "fork-augmented training. Use --recluster for "
                             "combined clustering.")
    parser.add_argument("--tool-arg-beta", type=float, default=0.0,
                        help="Tool-argument similarity penalty exponent. "
                             "When > 0, penalizes similarity between steps "
                             "calling the same tool with different arguments. "
                             "0.0 = disabled (default), 2.0 recommended.")
    parser.add_argument("--levels", default=None,
                        help="Comma-separated list of levels to evaluate "
                             "(e.g. level0_markov,level4_gru). Default: all.")
    parser.add_argument("--fasttext-ngrams", type=int, default=None,
                        help="Override wordNgrams for level3_fasttext "
                             "(default: 5)")
    parser.add_argument("--heldout-eval", action="store_true",
                        help="Train PRMs on --data-dir, evaluate on --heldout-dir")
    parser.add_argument("--augmented-heldout", action="store_true",
                        help="Train PRMs on --data-dir + --extra-train-dir, "
                             "evaluate on --heldout-dir")
    parser.add_argument("--extra-train-dir", default=None,
                        help="Directory with additional training rollouts + embeddings "
                             "(for --augmented-heldout or --combined-loo)")
    parser.add_argument("--combined-loo", action="store_true",
                        help="LOO-CV on combined --data-dir + --heldout-dir "
                             "(+ optional --extra-train-dir) rollouts")
    parser.add_argument("--heldout-dir", default=None,
                        help="Directory containing held-out rollouts + embeddings "
                             "(required for --heldout-eval and --combined-loo)")
    parser.add_argument("--recluster", action="store_true",
                        help="With --combined-loo: re-cluster on all combined "
                             "embeddings instead of NN-assigning to original clusters")
    args = parser.parse_args()

    ns = args.noise_strategy
    if args.out_dir:
        out = args.out_dir
    elif ns == "outlier_bin":
        out = str(Path(args.data_dir) / "analysis" / "prm_noise_bin")
    else:
        out = str(Path(args.data_dir) / "analysis" / "prm")

    beta = args.tool_arg_beta
    level_filter = set(args.levels.split(",")) if args.levels else None

    if args.fasttext_ngrams is not None:
        LEVELS["level3_fasttext"] = (
            FastTextPRM, {"wordNgrams": args.fasttext_ngrams, "dim": 20})

    if args.augmented_heldout:
        if not args.heldout_dir:
            parser.error("--augmented-heldout requires --heldout-dir")
        if not args.extra_train_dir:
            parser.error("--augmented-heldout requires --extra-train-dir")
        run_augmented_heldout_eval(
            args.data_dir, args.extra_train_dir, args.heldout_dir, out,
            min_cluster_size=args.mcs,
            noise_strategy=ns,
            tool_arg_beta=beta,
            level_filter=level_filter,
        )
    elif args.heldout_eval:
        if not args.heldout_dir:
            parser.error("--heldout-eval requires --heldout-dir")
        run_heldout_eval(
            args.data_dir, args.heldout_dir, out,
            min_cluster_size=args.mcs,
            noise_strategy=ns,
            tool_arg_beta=beta,
            level_filter=level_filter,
        )
    elif args.combined_loo:
        if not args.heldout_dir:
            parser.error("--combined-loo requires --heldout-dir")
        run_combined_loo(
            args.data_dir, args.heldout_dir, out,
            min_cluster_size=args.mcs,
            noise_strategy=ns,
            tool_arg_beta=beta,
            level_filter=level_filter,
            recluster=args.recluster,
            extra_train_dir=args.extra_train_dir,
        )
    elif args.fork_prm:
        run_fork_prm_experiment(
            args.data_dir, out,
            min_cluster_size=args.mcs,
            noise_strategy=ns,
            tool_arg_beta=beta,
            level_filter=level_filter,
        )
    elif args.fork_loo:
        run_fork_loo_experiment(
            args.data_dir, out,
            fork_data_dir=args.fork_data_dir,
            min_cluster_size=args.mcs,
            noise_strategy=ns,
            tool_arg_beta=beta,
            level_filter=level_filter,
            recluster=args.recluster,
        )
    elif args.fork_heldout:
        if not args.heldout_dir:
            parser.error("--fork-heldout requires --heldout-dir")
        run_fork_heldout_eval(
            args.data_dir, args.heldout_dir, out,
            fork_data_dir=args.fork_data_dir,
            min_cluster_size=args.mcs,
            noise_strategy=ns,
            tool_arg_beta=beta,
            level_filter=level_filter,
            recluster=args.recluster,
        )
    elif args.fork_combined_loo:
        if not args.heldout_dir:
            parser.error("--fork-combined-loo requires --heldout-dir")
        run_fork_combined_loo(
            args.data_dir, args.heldout_dir, out,
            fork_data_dir=args.fork_data_dir,
            min_cluster_size=args.mcs,
            noise_strategy=ns,
            tool_arg_beta=beta,
            level_filter=level_filter,
            recluster=args.recluster,
        )
    elif args.fork_recluster:
        recluster_fork_strategies(
            args.data_dir, out,
            min_cluster_size=args.mcs,
            noise_strategy=ns,
            tool_arg_beta=beta,
        )
    elif args.fork:
        if args.diagnose_only:
            diagnose_fork_points(
                args.data_dir, out,
                min_cluster_size=args.mcs,
                noise_strategy=ns,
            )
        else:
            run_fork_experiment(
                args.data_dir, out,
                min_cluster_size=args.mcs,
                noise_strategy=ns,
            )
    elif args.biased_subsample:
        if args.diagnose_only:
            diagnose_branch_weights(
                args.data_dir, out,
                min_cluster_size=args.mcs,
                noise_strategy=ns,
            )
        else:
            run_biased_subsample_experiment(
                args.data_dir, out,
                min_cluster_size=args.mcs,
                noise_strategy=ns,
            )
    elif args.subsample_hard:
        mcs_vals = args.mcs_sweep or [args.mcs]
        run_subsample_hard_experiment(
            args.data_dir, out,
            min_cluster_size=args.mcs,
            noise_strategy=ns,
            mcs_values=mcs_vals,
        )
    elif args.convergence:
        run_subsample_convergence(
            args.data_dir, out,
            min_cluster_size=args.mcs,
            noise_strategy=ns,
        )
    elif args.subsample:
        mcs_vals = args.mcs_sweep or [args.mcs]
        run_subsample_experiment(
            args.data_dir, out,
            min_cluster_size=args.mcs,
            noise_strategy=ns,
            mcs_values=mcs_vals,
        )
    elif args.manifest:
        tasks = load_all_tasks(args.data_dir, min_cluster_size=args.mcs,
                               noise_strategy=ns, tool_arg_beta=beta)
        build_branching_manifest(tasks, out, n_targets=args.n_targets)
    elif args.disagree:
        tasks = load_all_tasks(args.data_dir, min_cluster_size=args.mcs,
                               noise_strategy=ns, tool_arg_beta=beta)
        analyze_disagreements(tasks, out)
    elif args.step_bon:
        tasks = load_all_tasks(args.data_dir, min_cluster_size=args.mcs,
                               noise_strategy=ns, tool_arg_beta=beta)
        compute_step_level_bon(tasks, out)
    elif args.complexity:
        analyze_complexity(args.data_dir, out, min_cluster_size=args.mcs,
                           noise_strategy=ns)
    elif args.plots_only:
        replot(out)
    else:
        run_all(data_dir=args.data_dir, out_dir=out, min_cluster_size=args.mcs,
                noise_strategy=ns, tool_arg_beta=beta, level_filter=level_filter)


if __name__ == "__main__":
    main()
