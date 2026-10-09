#!/usr/bin/env python3
"""Download Spark Studio model packs into ComfyUI's models folder.

Usage:
  ./spark-studio models --list
  ./spark-studio models image edit        # one or more packs
  ./spark-studio models all

Downloads resume if interrupted; just run the same command again.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "app" / "models.json"


def load_config():
    cfg = {}
    env_file = HERE / "config.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"')
    return cfg


def main():
    packs = json.loads(MANIFEST.read_text())["packs"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("packs", nargs="*", help=f"packs to download: {', '.join(packs)} or all")
    ap.add_argument("--comfy-dir", help="ComfyUI folder (defaults to COMFY_DIR in config.env)")
    ap.add_argument("--list", action="store_true", help="show packs and what is already downloaded")
    ap.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    args = ap.parse_args()

    comfy = Path(args.comfy_dir or os.environ.get("COMFY_DIR") or load_config().get("COMFY_DIR", "")).expanduser()
    if not (comfy / "main.py").exists():
        sys.exit(f"Can't find ComfyUI at '{comfy}'. Pass --comfy-dir /path/to/ComfyUI")
    models = comfy / "models"

    def target(f):
        return models / f["dir"] / Path(f["path"]).name

    if args.list or not args.packs:
        for key, p in packs.items():
            have = sum(target(f).exists() for f in p["files"])
            size = sum(f["gb"] for f in p["files"])
            state = "installed" if have == len(p["files"]) else f"{have}/{len(p['files'])} files"
            print(f"  {key:6} {p['label']:42} ~{size:5.1f} GB   {state}")
        if not args.packs:
            print("\nExample: ./spark-studio models image   (or: all)")
        return

    wanted = list(packs) if "all" in args.packs else args.packs
    bad = [w for w in wanted if w not in packs]
    if bad:
        sys.exit(f"Unknown pack(s): {', '.join(bad)}. Choose from: {', '.join(packs)}, all")

    todo, seen = [], set()
    for w in wanted:
        for f in packs[w]["files"]:
            t = target(f)
            if t in seen or t.exists():
                continue
            seen.add(t)
            todo.append(f)

    if not todo:
        print("Everything for those packs is already downloaded.")
        return

    need = sum(f["gb"] for f in todo)
    free = shutil.disk_usage(models).free / 1e9
    print(f"Will download {len(todo)} files, about {need:.0f} GB (free space: {free:.0f} GB).")
    for f in todo:
        print(f"  {f['gb']:5.1f} GB  {f['dir']}/{Path(f['path']).name}")
    if free < need + 5:
        sys.exit("Not enough free disk space.")
    if not args.yes and sys.stdin.isatty():
        if input("Continue? [Y/n] ").strip().lower() not in ("", "y", "yes"):
            return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is missing. Run it as: ./spark-studio models ...")

    staging = models / ".spark-studio-staging"
    for i, f in enumerate(todo, 1):
        t = target(f)
        print(f"\n[{i}/{len(todo)}] {t.name}")
        t.parent.mkdir(parents=True, exist_ok=True)
        local = hf_hub_download(repo_id=f["repo"], filename=f["path"], local_dir=staging)
        os.replace(local, t)
        print(f"  saved to {t}")
    shutil.rmtree(staging, ignore_errors=True)
    print("\nDone. Refresh Spark Studio in your browser to use the new packs.")


if __name__ == "__main__":
    main()
