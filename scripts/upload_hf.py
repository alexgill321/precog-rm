#!/usr/bin/env python3
"""Maintainer tool: upload the release to HuggingFace.

Dataset repo agill32/precog-rm (browsable parts, see data/README.md):
  rollouts/, embeddings/        from the tree built by scripts/build_release.py (--tree)
  grpo/, caches/, ablations/, meta/   from the data root ($PRECOG_DATA)
Model repo agill32/precog-rm-checkpoints: ckpts/ re-foldered as sweep/, design_study/, prefix/, grpo_controller.pt.

Safety: credential-like files and scratch dirs are never copied (EXCLUDE_*),
every staged part is scanned for secret patterns before upload, and
file-heavy ablation trees ship as tarballs. Repos are created PRIVATE.

    python scripts/upload_hf.py --part rollouts --benchmark tau2 --tree ../hf --staging ../staging2
    python scripts/upload_hf.py --part embeddings grpo caches ablations meta --tree ../hf --staging ../staging2
    python scripts/upload_hf.py --part checkpoints --staging ../staging2
    python scripts/upload_hf.py --part rollouts --dry-run ...     # materialize + scan only
"""
import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from _paths import DATA, REPO, DATASETS, POLICY_DIR

ORG = "agill32"
DATASET_REPO = f"{ORG}/precog-rm"
MODEL_REPO = f"{ORG}/precog-rm-checkpoints"
CARDS = REPO / "data" / "cards"

EXCLUDE_DIRS = {"_bfcl_tmp", "__pycache__", ".cache", ".git", ".ipynb_checkpoints", ".file_locks"}
EXCLUDE_FILE = re.compile(r"^\.env(\..*)?$|^\.DS_Store$|\.(pem|key|p12)$|credential|secret"
                          r"|^train\.log$|^tinker_session\.txt$", re.I)
SECRET_PAT = re.compile(rb"sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|tml-[A-Za-z0-9_-]{8,}|tinker_[A-Za-z0-9_-]{10,}"
                        rb"|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}"
                        rb"|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|\"private_key\"|sk-or-v1-[a-f0-9]{20,}|sk-ant-[A-Za-z0-9_-]{20,}"
                        rb"|API_KEY *[=:] *[\"']?[A-Za-z0-9][A-Za-z0-9_-]{15,}"
                        rb"|Bearer (?=[A-Za-z0-9._-]*[a-z0-9])[A-Za-z0-9._-]{20,}")
TEXT_EXT = {".json", ".jsonl", ".txt", ".log", ".yaml", ".yml", ".csv", ".md", ".cfg", ".ini", ".toml", ".py", ".sh"}


def scan_secrets(root: Path, max_bytes=200 * 2**20):
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


def link_tree(src: Path, dst: Path):
    """Hard-link (or copy) a tree, following symlinks, skipping excluded names."""
    for root, dirs, files in os.walk(src, followlinks=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            if EXCLUDE_FILE.search(f):
                continue
            d = dst / rel / f
            if d.exists():
                continue
            s = os.path.realpath(Path(root) / f)
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)


def make_tar(src_dir: Path, out: Path):
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["tar", "-czhf", str(out)]
    for d in EXCLUDE_DIRS:
        cmd += ["--exclude", d]
    for pat in (".env", ".env.*", ".DS_Store"):
        cmd += ["--exclude", pat]
    cmd += ["-C", str(src_dir.parent), src_dir.name]
    print(f"    tar {out} <- {src_dir}", flush=True)
    subprocess.run(cmd, check=True)


def stage_part(part, tree: Path, stage_root: Path, benchmarks, policies):
    """Materialize one part under <stage_root>/dataset/<part>/ (or /model/ for checkpoints).
    upload_large_folder pushes a whole folder as the repo root, so the repo-root
    folder is returned together with the part subfolder to scan."""
    stage = stage_root / ("model" if part == "checkpoints" else "dataset")
    if part in ("rollouts", "embeddings"):
        for b in benchmarks:
            for pol in policies:
                src = tree / part / b / pol
                if src.exists():
                    link_tree(src, stage / part / b / pol)
        if part == "embeddings":
            for sub in ("reasoning", "grpo_replay"):
                if (DATA / "embeddings" / sub).exists():
                    link_tree(DATA / "embeddings" / sub, stage / "embeddings" / sub)
    elif part == "grpo":
        for sub in ("runs", "replay", "eval"):
            link_tree(DATA / "grpo" / sub, stage / "grpo" / sub)
    elif part == "caches":
        link_tree(DATA / "caches" / "mmap", stage / "caches" / "mmap")
        shutil.copy2(DATA / "caches" / "task_splits.json", stage / "caches" / "task_splits.json")
    elif part == "ablations":
        ac = DATA / "ablations" / "agentclinic"
        for pol in sorted(p.name for p in (ac / "notools").iterdir() if p.is_dir()):
            make_tar(ac / "notools" / pol, stage / "ablations" / "agentclinic" / "notools" / f"{pol}.tar.gz")
        make_tar(ac / "terra", stage / "ablations" / "agentclinic" / "terra.tar.gz")
        link_tree(ac / "resample", stage / "ablations" / "agentclinic" / "resample")
    elif part == "meta":
        link_tree(DATA / "meta", stage / "meta")
    elif part == "checkpoints":
        c = DATA / "ckpts"
        link_tree(c / "precog", stage / "checkpoints" / "sweep")
        link_tree(c / "ablations", stage / "checkpoints" / "design_study")
        for p in (c / "prefix").iterdir():
            if p.name == "grpo_controller_model_prefix.pt":
                (stage / "checkpoints").mkdir(parents=True, exist_ok=True)
                if not (stage / "checkpoints" / "grpo_controller.pt").exists():
                    os.link(os.path.realpath(p), stage / "checkpoints" / "grpo_controller.pt")
            else:
                (stage / "checkpoints" / "prefix").mkdir(parents=True, exist_ok=True)
                if not (stage / "checkpoints" / "prefix" / p.name).exists():
                    os.link(os.path.realpath(p), stage / "checkpoints" / "prefix" / p.name)
    return stage, (stage / "checkpoints" if part == "checkpoints" else stage / part)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", nargs="+", required=True,
                    choices=["rollouts", "embeddings", "grpo", "caches", "ablations", "meta", "checkpoints"])
    ap.add_argument("--tree", type=Path, default=REPO.parent / "hf", help="output of build_release.py")
    ap.add_argument("--staging", type=Path, default=None)
    ap.add_argument("--benchmark", nargs="*", default=DATASETS, choices=DATASETS)
    ap.add_argument("--policy", nargs="*", default=list(POLICY_DIR.values()), choices=list(POLICY_DIR.values()))
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stage = args.staging or Path(tempfile.mkdtemp(prefix="precog_upload_"))
    from huggingface_hub import HfApi
    api = HfApi()
    for part in args.part:
        root, folder = stage_part(part, args.tree, stage, args.benchmark, args.policy)
        repo, kind = (MODEL_REPO, "model") if part == "checkpoints" else (DATASET_REPO, "dataset")
        card = CARDS / ("checkpoints.md" if part == "checkpoints" else "dataset.md")
        print(f"[{part}] scanning {folder} ...", flush=True)
        scan_secrets(folder)
        n = sum(1 for p in folder.rglob("*") if p.is_file())
        if args.dry_run:
            print(f"[{part}] dry run: {n} files staged at {folder}"); continue
        api.create_repo(repo, repo_type=kind, private=not args.public, exist_ok=True)
        print(f"[{part}] uploading {n} files ({root}) -> {repo}", flush=True)
        api.upload_large_folder(repo_id=repo, repo_type=kind, folder_path=str(root))
        if card.exists():
            api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo, repo_type=kind,
                            commit_message="dataset card")
        print(f"[{part}] done: https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}")


if __name__ == "__main__":
    main()
