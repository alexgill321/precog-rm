#!/usr/bin/env python3
"""Embed AgentClinic medical-dialogue rollouts into action/response embeddings.

AgentClinic rollouts are doctor/patient(/measurement) dialogues. We treat each
**doctor turn as one step**:
  - action   = the doctor's utterance (question / ``REQUEST TEST`` / ``DIAGNOSIS READY``)
  - response = the following environment reply (patient answer or measurement result);
               the final doctor turn with no reply gets ``<end>``.

Output mirrors the tau2 convention consumed by ``src.prm.paired_action_response``
and the ``src.analysis`` suite::

    $PRECOG_DATA/agentclinic_embeddings/<source>/
      embeddings/action/{scenario_stem}.npz     # keys r{rollout}_s{step}, shape (1, 3072)
      embeddings/response/{scenario_stem}.npz
      manifest.json                              # per-scenario, per-rollout rewards + metadata

Rewards come from the rollout ``correct`` bool (1.0 / 0.0).

Usage::

    python -m src.prm.embed_agentclinic                       # dry-run, all sources
    python -m src.prm.embed_agentclinic --source gpt5mini_medqa --dry-run
    python -m src.prm.embed_agentclinic --apply               # embed all sources
    python -m src.prm.embed_agentclinic --source deepseek_nejm --apply
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
REPO = Path(__file__).resolve().parents[2]            # precog-rm root
sys.path.insert(0, str(REPO))
from src.prm.embeddings import task_id_to_filename  # noqa: E402

DATA = Path(os.environ.get("PRECOG_DATA", REPO / "data"))
AC_RAW = DATA / "rollouts"                             # rollouts/agentclinic/<policy>/<scenario>/rollout_NN.json (public layout)
OUT_ROOT = DATA / "agentclinic_embeddings"

MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
BATCH_SIZE = 2048
MAX_TEXT_CHARS = 8_000  # text-embedding-3-large has an 8192-token limit

# Doctor turns are the "agent"; everything else is the environment reply.
_DOCTOR_ROLE = "doctor"

SOURCES = {
    "deepseek_medqa": {
        "rollouts_dir": AC_RAW / "agentclinic" / "deepseek-v3.2",
        "model": "vertex-deepseek-v3.2",
        "dataset": "MedQA_Ext",
    },
    "gpt5mini_medqa": {
        "rollouts_dir": AC_RAW / "agentclinic" / "gpt5mini",
        "model": "gpt-5-mini-2025-08-07",
        "dataset": "MedQA_Ext",
    },
    # Qwen doctor turns carry a separate `reasoning` field (not part of
    # `content`), which the extraction below never reads — embeddings cover
    # only the spoken utterance / extracted action, same as other sources.
    "qwen_medqa": {
        "rollouts_dir": AC_RAW / "agentclinic" / "qwen3-80b",
        "model": "vertex-qwen3-80b",
        "dataset": "MedQA_Ext",
    },
    "gpt56terra_medqa": {
        "rollouts_dir": AC_RAW / "agentclinic" / "gpt5.6-terra",
        "model": "gpt-5.6-terra",
        "dataset": "MedQA_Ext",
    },
}

_ROLLOUT_RE = re.compile(r"scenario_(\d+)_rollout_(\d+)\.json$")


def _truncate(text: str) -> str:
    return text[:MAX_TEXT_CHARS] if len(text) > MAX_TEXT_CHARS else text


def _clean(content) -> str:
    text = str(content).strip() if content is not None else ""
    return _truncate(text) if text else "<empty>"


def _extract_doctor_action(content) -> str:
    """Reduce a doctor turn to the ACTION it takes, using the same dispatch
    the AgentClinic env uses (case-insensitive markers).

      - "diagnosis ready: X"  -> "DIAGNOSIS: X"      (the diagnosis)
      - "request test: X"     -> "REQUEST TEST: X"   (the test requested)
      - otherwise             -> the question asked (leading "Doctor:" stripped)

    This strips conversational preamble/reasoning so the embedding reflects the
    action (which test / which diagnosis / what was asked) rather than wording.
    """
    text = str(content).strip() if content is not None else ""
    if not text:
        return "<empty>"
    low = text.lower()

    def _after(marker: str) -> str:
        i = low.find(marker)
        return text[i + len(marker):].strip(" :-\n\t") if i >= 0 else ""

    if "diagnosis ready" in low:
        diag = _after("diagnosis ready")
        return _truncate("DIAGNOSIS: " + diag) if diag else "DIAGNOSIS: <unspecified>"
    if "request test" in low:
        test = _after("request test")
        # keep only the first line/sentence of the test spec
        test = re.split(r"[\n.]", test, 1)[0].strip() if test else ""
        return _truncate("REQUEST TEST: " + test) if test else "REQUEST TEST: <unspecified>"
    # question to the patient — drop a leading speaker prefix if present
    q = re.sub(r"^(doctor|dr\.?\s*agent)\s*:\s*", "", text, flags=re.IGNORECASE)
    return _truncate(q.strip()) or "<empty>"


def _extract_action_response(
    history: list[dict], action_mode: str = "full",
    drop_final_diagnosis: bool = False,
) -> tuple[list[str], list[str]]:
    """Map a dialogue history to per-doctor-step (action, response) texts.

    A step is opened by each doctor turn. Its response is the immediately
    following non-doctor turn; a trailing doctor turn (or a doctor turn
    followed by another doctor turn) gets ``<end>``.

    action_mode:
      - "full" (v1): embed the doctor's raw utterance.
      - "action" (v2): embed only the extracted action (diagnosis / test /
        question) via the env's dispatch logic.

    drop_final_diagnosis (nodiag ablation): if the rollout ends in a
    ``DIAGNOSIS READY`` turn, drop that terminal step entirely. Removes the
    outcome-revealing final action so separability reflects the prefix, not the
    stated answer. Requires action_mode="action".
    """
    actions: list[str] = []
    responses: list[str] = []
    n = len(history)
    for i, turn in enumerate(history):
        if turn.get("role") != _DOCTOR_ROLE:
            continue
        content = turn.get("content")
        if action_mode == "action":
            actions.append(_extract_doctor_action(content))
        else:
            actions.append(_clean(content))
        if i + 1 < n and history[i + 1].get("role") != _DOCTOR_ROLE:
            responses.append(_clean(history[i + 1].get("content")))
        else:
            responses.append("<end>")
    if drop_final_diagnosis and actions and actions[-1].startswith("DIAGNOSIS"):
        actions.pop()
        responses.pop()
    return actions, responses


def _discover_scenarios(rollouts_dir: Path) -> list[dict]:
    """Group rollout files by scenario id."""
    by_scenario: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for fp in rollouts_dir.glob("scenario_*_rollout_*.json"):
        m = _ROLLOUT_RE.search(fp.name)
        if not m:
            continue
        sid, rid = int(m.group(1)), int(m.group(2))
        by_scenario[sid].append((rid, fp))

    scenarios = []
    for sid in sorted(by_scenario):
        rollouts = sorted(by_scenario[sid])  # by rid
        scenarios.append({
            "scenario_id": sid,
            "stem": f"scenario_{sid:04d}",
            "rollouts": rollouts,       # list[(rid, path)]
            "n_rollouts": len(rollouts),
        })
    return scenarios


def _embed_batch(client, texts: list[str]) -> list[np.ndarray]:
    resp = client.embeddings.create(input=texts, model=MODEL)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]


def _embed_scenario(client, scenario: dict, action_mode: str = "full",
                    drop_final_diagnosis: bool = False) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[int, float]
]:
    """Return (action_arrays, response_arrays, rewards) for one scenario.

    Texts are deduplicated across all rollouts/steps of the scenario before
    the API call; each NPZ key r{rid}_s{step} stores a (1, EMBED_DIM) row.
    """
    unique: dict[str, int] = {}
    text_list: list[str] = []

    def _intern(text: str) -> int:
        idx = unique.get(text)
        if idx is None:
            idx = len(text_list)
            unique[text] = idx
            text_list.append(text)
        return idx

    # structure[rid] = (action_idx_per_step, response_idx_per_step)
    structure: list[tuple[int, list[int], list[int]]] = []
    rewards: dict[int, float] = {}

    for rid, fp in scenario["rollouts"]:
        with open(fp) as fh:
            raw = json.load(fh)
        rewards[rid] = 1.0 if raw.get("correct") else 0.0
        actions, responses = _extract_action_response(
            raw.get("history", []), action_mode=action_mode,
            drop_final_diagnosis=drop_final_diagnosis)
        a_idx = [_intern(t) for t in actions]
        r_idx = [_intern(t) for t in responses]
        structure.append((rid, a_idx, r_idx))

    # Embed all unique non-empty texts.
    zero = np.zeros(EMBED_DIM, dtype=np.float32)
    empty_idx = unique.get("<empty>")
    embeddable = [(i, t) for i, t in enumerate(text_list) if t != "<empty>"]
    vecs: dict[int, np.ndarray] = {}
    for start in range(0, len(embeddable), BATCH_SIZE):
        batch = embeddable[start : start + BATCH_SIZE]
        out = _embed_batch(client, [t for _, t in batch])
        for (idx, _), v in zip(batch, out):
            vecs[idx] = v

    def _row(text_idx: int) -> np.ndarray:
        if text_idx == empty_idx:
            return zero
        return vecs.get(text_idx, zero)

    action_arrays: dict[str, np.ndarray] = {}
    response_arrays: dict[str, np.ndarray] = {}
    for rid, a_idx, r_idx in structure:
        for s, ti in enumerate(a_idx):
            action_arrays[f"r{rid}_s{s}"] = _row(ti)[None, :]
        for s, ti in enumerate(r_idx):
            response_arrays[f"r{rid}_s{s}"] = _row(ti)[None, :]

    return action_arrays, response_arrays, rewards


def _load_api_key() -> str | None:
    """OPENAI_API_KEY from the environment."""
    return os.environ.get("OPENAI_API_KEY")


def _run_source(name: str, cfg: dict, apply: bool, action_mode: str = "full",
                drop_final_diagnosis: bool = False) -> None:
    rollouts_dir = cfg["rollouts_dir"]
    out_name = name if action_mode == "full" else f"{name}_v2"
    if drop_final_diagnosis:
        out_name += "_nodiag"
    out_dir = OUT_ROOT / out_name
    emb_dir = out_dir / "embeddings"
    act_dir = emb_dir / "action"
    res_dir = emb_dir / "response"

    print(f"\n===== {out_name} (action_mode={action_mode}) =====")
    if not rollouts_dir.exists():
        print(f"  SKIP: rollouts dir not found: {rollouts_dir}")
        return

    scenarios = _discover_scenarios(rollouts_dir)
    total_rollouts = sum(s["n_rollouts"] for s in scenarios)
    print(f"  {rollouts_dir.relative_to(REPO)}")
    print(f"  {len(scenarios)} scenarios, {total_rollouts} rollouts")

    to_embed, done = [], 0
    for s in scenarios:
        stem = task_id_to_filename(s["stem"])
        act_npz = act_dir / f"{stem}.npz"
        res_npz = res_dir / f"{stem}.npz"
        if act_npz.exists() and res_npz.exists():
            try:
                with np.load(res_npz) as npz:
                    rids = {int(k.split("_")[0][1:]) for k in npz.files}
                with np.load(act_npz) as npz:  # verify action side too
                    _ = npz.files
                if len(rids) >= s["n_rollouts"]:
                    done += 1
                    continue
            except (zipfile.BadZipFile, OSError, ValueError):
                # corrupt/partial NPZ (e.g. from an interrupted write) — re-embed
                print(f"    re-embedding {stem}: unreadable NPZ")
        to_embed.append(s)
    print(f"  already embedded: {done} | need embedding: {len(to_embed)}")

    # Report step-text volume (helps size cost before spending).
    if to_embed and not apply:
        sample = to_embed[: min(20, len(to_embed))]
        step_texts = 0
        for s in sample:
            for _rid, fp in s["rollouts"]:
                hist = json.load(open(fp)).get("history", [])
                a, r = _extract_action_response(
                    hist, action_mode=action_mode,
                    drop_final_diagnosis=drop_final_diagnosis)
                step_texts += len(a) + len(r)
        avg = step_texts / max(sum(x["n_rollouts"] for x in sample), 1)
        est = int(avg * total_rollouts)
        print(f"  ~{avg:.0f} step-texts/rollout → ~{est:,} pre-dedup texts for this source")

    if not apply:
        return

    from openai import OpenAI

    key = _load_api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=key)

    act_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    manifest_scenarios = []
    total_keys = 0
    for i, s in enumerate(to_embed):
        t0 = time.time()
        stem = task_id_to_filename(s["stem"])
        act_arrays, res_arrays, rewards = _embed_scenario(
            client, s, action_mode, drop_final_diagnosis)
        np.savez_compressed(str(act_dir / f"{stem}.npz"), **act_arrays)
        np.savez_compressed(str(res_dir / f"{stem}.npz"), **res_arrays)
        total_keys += len(act_arrays) + len(res_arrays)
        manifest_scenarios.append({
            "scenario_id": s["scenario_id"],
            "stem": stem,
            "n_rollouts": s["n_rollouts"],
            "rewards": {str(k): v for k, v in sorted(rewards.items())},
        })
        print(f"  [{i+1}/{len(to_embed)}] {stem}  {s['n_rollouts']} rollouts  "
              f"{len(act_arrays)} steps  {time.time()-t0:.1f}s")

    # Merge manifest with any prior run (resume-safe).
    manifest_path = out_dir / "manifest.json"
    prior = {}
    if manifest_path.exists():
        prior = {e["scenario_id"]: e for e in json.load(open(manifest_path)).get("scenarios", [])}
    for e in manifest_scenarios:
        prior[e["scenario_id"]] = e
    manifest = {
        "source": out_name,
        "action_mode": action_mode,
        "drop_final_diagnosis": drop_final_diagnosis,
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "rollouts_dir": str(rollouts_dir.relative_to(REPO)),
        "embed_model": MODEL,
        "embed_dim": EMBED_DIM,
        "n_scenarios": len(prior),
        "scenarios": [prior[k] for k in sorted(prior)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {len(prior)} scenarios to manifest; {total_keys} new NPZ keys")


# ── AgentClinic-specific discovery for downstream analysis ────────────────────


def discover_agentclinic_tasks(source: str) -> list[dict]:
    """Yield task dicts for the analysis suite: one per scenario, with the
    action NPZ path and per-rollout rewards (keyed by rollout id).

    Mirrors the contract of ``src.prm.thought_dag._discover_tasks`` closely
    enough that callers can use ``load_paired_embeddings(npz_path, response_npz)``.
    """
    out_dir = OUT_ROOT / source
    manifest = json.load(open(out_dir / "manifest.json"))
    tasks = []
    for e in manifest["scenarios"]:
        stem = e["stem"]
        act = out_dir / "embeddings" / "action" / f"{stem}.npz"
        res = out_dir / "embeddings" / "response" / f"{stem}.npz"
        if not (act.exists() and res.exists()):
            continue
        tasks.append({
            "task_label": stem,
            "npz_path": str(act),
            "response_npz_path": str(res),
            "rewards": {int(k): v for k, v in e["rewards"].items()},
        })
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=list(SOURCES), default=None,
                        help="Embed a single source (default: all).")
    parser.add_argument("--apply", action="store_true",
                        help="Actually call the embedding API (default: dry-run).")
    parser.add_argument("--variant", choices=["v1", "v2", "nodiag"], default="v1",
                        help="v1 (full doctor utterance), v2 (extracted "
                             "action: diagnosis/test/question), or nodiag "
                             "(v2 with the terminal diagnosis step dropped — "
                             "leakage ablation).")
    args = parser.parse_args()

    drop_final_diagnosis = args.variant == "nodiag"
    action_mode = "full" if args.variant == "v1" else "action"
    names = [args.source] if args.source else list(SOURCES)
    print(f"AgentClinic embedding — {'APPLY' if args.apply else 'DRY RUN'} — "
          f"variant={args.variant} — sources: {names}")
    print(f"Output root: {OUT_ROOT}")
    for name in names:
        _run_source(name, SOURCES[name], args.apply, action_mode,
                    drop_final_diagnosis)
    if not args.apply:
        print("\nDRY RUN — re-run with --apply to embed.")


if __name__ == "__main__":
    main()
