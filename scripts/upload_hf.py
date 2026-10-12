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


import re

# never published: credential files, harness scratch dirs, caches, OS cruft
EXCLUDE_DIRS = {"_bfcl_tmp", "__pycache__", ".cache", ".git", ".ipynb_checkpoints"}
EXCLUDE_FILE = re.compile(r"^\.env(\..*)?$|^\.DS_Store$|\.(pem|key|p12)$|credential|secret", re.I)
SECRET_PAT = re.compile(rb"sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|tinker_[A-Za-z0-9_-]{10,}|tk-[A-Za-z0-9_-]{16,}"
                        rb"|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}"
                        rb"|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|\"private_key\"|sk-or-v1-[a-f0-9]{20,}"
                        rb"|API_KEY *[=:] *[\"']?[A-Za-z0-9][A-Za-z0-9_-]{15,}|Bearer [A-Za-z0-9._-]{20,}")
TEXT_EXT = {".json", ".jsonl", ".txt", ".log", ".yaml", ".yml", ".csv", ".md", ".cfg", ".ini", ".toml", ".py", ".sh"}


def scan_secrets(root: Path, max_bytes=50 * 2**20):
    """Abort the upload if any text file under root matches a credential pattern."""
    hits = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in TEXT_EXT and p.stat().st_size <= max_bytes:
            with open(p, "rb") as fh:
                if SECRET_PAT.search(fh.read()):
                    hits.append(p)
                    if len(hits) >= 20:
                        break
    if hits:
        raise SystemExit("REFUSING TO UPLOAD: credential-like content in\n  " + "\n  ".join(map(str, hits)))


def materialize(src: Path, dst: Path):
    """Copy a (possibly symlinked) tree into dst following links; big files are
    hard-linked when on the same volume, else copied. Credential files and
    scratch directories are skipped (EXCLUDE_DIRS / EXCLUDE_FILE)."""
    for root, dirs, files in os.walk(src, followlinks=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            if EXCLUDE_FILE.search(f):
                continue
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
        print(f"[{st}] scanning {stage_root} for credentials ...", flush=True)
        scan_secrets(stage_root)
        print(f"[{st}] uploading {stage_root} -> {repo} ({kind})", flush=True)
        api.upload_large_folder(repo_id=repo, repo_type=kind, folder_path=str(stage_root))
        print(f"[{st}] done: https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}")


if __name__ == "__main__":
    main()
