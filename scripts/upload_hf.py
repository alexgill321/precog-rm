#!/usr/bin/env python3
"""Maintainer tool: upload the staged data root to the HuggingFace release repos.

Expects $PRECOG_DATA laid out as in data/README.md (symlinks are followed).
Creates the repos PRIVATE if they do not exist; flip them public from the Hub
UI after checking the dataset cards. Uses upload_large_folder (resumable,
multi-part) so a 25 GB stage can be interrupted and resumed.

    hf auth login                       # once
    python scripts/upload_hf.py --stage checkpoints
    python scripts/upload_hf.py --stage embeddings caches grpo rollouts
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

from _paths import DATA, REPO
from download import REPOS

# stage -> (subdirectories of DATA to upload, path_in_repo prefix)
STAGE_DIRS = {
    "embeddings": ["table1_data", "reasoning_emb", "grpo_replay"],
    "caches": ["caches"],
    "checkpoints": ["ckpts"],
    "grpo": ["runs", "replay", "eval"],
    "rollouts": ["rollouts"],
}
CARDS = REPO / "data" / "cards"


def materialize(src: Path, dst: Path):
    """Copy a (possibly symlinked) tree into dst following links; big files are
    hard-linked when on the same volume, else copied."""
    for root, dirs, files in os.walk(src, followlinks=True):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            s = Path(root) / f
            d = dst / rel / f
            if d.exists():
                continue
            try:
                os.link(os.path.realpath(s), d)
            except OSError:
                shutil.copy2(os.path.realpath(s), d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", choices=list(STAGE_DIRS), required=True)
    ap.add_argument("--staging", default=None, help="scratch dir for the materialized tree (default: temp)")
    ap.add_argument("--public", action="store_true", help="create repos public (default private)")
    args = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    for st in args.stage:
        repo, kind = REPOS[st]
        api.create_repo(repo, repo_type=kind, private=not args.public, exist_ok=True)
        stage_root = Path(args.staging or tempfile.mkdtemp(prefix="precog_upload_")) / st
        for sub in STAGE_DIRS[st]:
            src = DATA / sub
            if not src.exists():
                print(f"  [skip] {src} missing"); continue
            print(f"  materializing {sub} ...", flush=True)
            materialize(src, stage_root / sub)
        card = CARDS / f"{st}.md"
        if card.exists():
            shutil.copy2(card, stage_root / "README.md")
        print(f"[{st}] uploading {stage_root} -> {repo} ({kind})", flush=True)
        api.upload_large_folder(repo_id=repo, repo_type=kind, folder_path=str(stage_root))
        print(f"[{st}] done: https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}")


if __name__ == "__main__":
    main()
