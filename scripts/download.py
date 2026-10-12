#!/usr/bin/env python3
"""Download the data release from HuggingFace into $PRECOG_DATA (default <repo>/data).

Stages map to the pipeline entry points (see data/README.md for the layout):

  embeddings   table1_data/ (9 cells, ~25 GB), reasoning_emb/, grpo_replay/ (13 GB)   -> everything
  caches       caches/mmap (10 GB, float16 fast path; skips the HDBSCAN rebuild)
  checkpoints  ckpts/ (precog sweep, design-study and prefix checkpoints, ~0.5 GB)
  grpo         runs/ (21 official GRPO runs, ~35 GB), replay/, eval/
  rollouts     rollouts/ (raw trajectories, no-tools ablation, resample JSONs, meta lists)

    python scripts/download.py --stage checkpoints caches
    python scripts/download.py --stage embeddings --cells tau2_qwen bfclv3_deepseek
    python scripts/download.py --all
"""
import argparse

from _paths import DATA

ORG = "agill32"
REPOS = {
    "embeddings": (f"{ORG}/precog-rm-embeddings", "dataset"),
    "caches": (f"{ORG}/precog-rm-embeddings", "dataset"),
    "checkpoints": (f"{ORG}/precog-rm-checkpoints", "model"),
    "grpo": (f"{ORG}/precog-rm-grpo-logs", "dataset"),
    "rollouts": (f"{ORG}/precog-rm-rollouts", "dataset"),
}
PATTERNS = {
    "embeddings": ["table1_data/**", "reasoning_emb/**", "grpo_replay/**", "README.md"],
    "caches": ["caches/**"],
    "checkpoints": ["**"],
    "grpo": ["**"],
    "rollouts": ["**"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="*", choices=list(REPOS), default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cells", nargs="*", help="embeddings stage: restrict table1_data to these cells")
    ap.add_argument("--no-grpo-replay", action="store_true", help="embeddings stage: skip the 13 GB replay embeddings")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    stages = list(REPOS) if args.all else args.stage
    if not stages:
        ap.error("give --stage ... or --all")
    DATA.mkdir(parents=True, exist_ok=True)
    for st in stages:
        repo, kind = REPOS[st]
        pats = list(PATTERNS[st])
        if st == "embeddings":
            if args.cells:
                pats = [f"table1_data/{c}/**" for c in args.cells] + ["reasoning_emb/**", "README.md"]
                if not args.no_grpo_replay:
                    pats.append("grpo_replay/**")
            elif args.no_grpo_replay:
                pats.remove("grpo_replay/**")
        print(f"[{st}] {repo} -> {DATA}  patterns={pats}", flush=True)
        snapshot_download(repo_id=repo, repo_type=kind, local_dir=str(DATA), allow_patterns=pats)
    extract_tarballs(DATA)
    print("done:", DATA)


def extract_tarballs(root):
    """File-heavy directories ship as tarballs (see scripts/upload_hf.py:tar_rule);
    extract each next to itself once, then drop the archive."""
    import tarfile
    for tar in sorted(root.rglob("*.tar.gz")):
        marker = tar.with_suffix("").with_suffix(".extracted")
        if marker.exists():
            continue
        print(f"  extracting {tar.relative_to(root)}", flush=True)
        with tarfile.open(tar) as tf:
            tf.extractall(tar.parent, filter="data")
        marker.write_text("ok\n")
        tar.unlink()


if __name__ == "__main__":
    main()
