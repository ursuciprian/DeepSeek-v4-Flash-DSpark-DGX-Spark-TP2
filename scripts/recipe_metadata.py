#!/usr/bin/env python3
"""Resolve launcher/download metadata from a sparkrun recipe without executing it."""
import json
from pathlib import Path
import re
import shlex
import sys

import yaml

# Existing pinned downloads; never reuse a revision for a different model.
REVISIONS = {
    "RadixArk/Qwen3.8-27B-NVFP4": "52d1adc5f38aa5ebf099c29ed7025ba34cfbb854",
    "RadixArk/Qwen3.8-Flash-Next-NVFP4": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
    "z-lab/Qwen3.8-27B-DFlash2": "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
}


def render(path):
    root = Path(__file__).resolve().parents[1]
    text = Path(path).read_text().replace(f"/home/nvidia/GEN-AI/{root.name}", str(root))
    if True:  # recipes written against earlier clone paths still resolve here
        for old in ("/home/nvidia/GEN-AI/qwen38-opt", "/home/nvidia/GEN-AI/flashnext",
                    "/home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2"):
            text = text.replace(old, str(root))
    recipe = yaml.safe_load(text)
    for volume in recipe.get("executor_config", {}).get("volumes", []):
        source = Path(volume.split(":", 1)[0])
        if source.is_relative_to(root) and not source.exists():
            raise ValueError(f"recipe references missing file: {source}")
    return text


def metadata(path, engine):
    recipe = yaml.safe_load(render(path))
    defaults = recipe.get("defaults", {})
    runtime = recipe.get("runtime") or engine
    if engine not in ("vllm", "sglang") or not isinstance(runtime, str) or not runtime.startswith(engine):
        raise ValueError(f"recipe runtime {runtime!r} does not match {engine!r}")
    model = recipe["model"]
    command = recipe["command"]
    tokens = shlex.split(command)
    def flag(name, fallback):
        if name not in tokens:
            return fallback
        index = tokens.index(name) + 1
        if index == len(tokens) or tokens[index].startswith("--"):
            raise ValueError(f"missing value for {name}")
        return tokens[index].format(model=model, **defaults)
    revision = flag("--revision", defaults.get("revision", recipe.get("revision", REVISIONS.get(model, ""))))
    served = flag("--served-model-name", defaults.get("served_model_name", model))
    port = int(flag("--port", defaults.get("port", 8000)))
    if not 1 <= port <= 65535:
        raise ValueError("recipe port must be between 1 and 65535")
    draft, draft_revision = "", ""
    if engine == "vllm":
        match = re.search(r"--speculative-config\s+(['\"])(.*?)\1", command, re.S)
        if match:
            config = json.loads(match[2].format(**defaults))
            if config.get("method") not in ("mtp", "ngram"):
                draft = config.get("model", "")
                draft_revision = config.get("revision", "")
    elif "--speculative-draft-model-path" in command:
        draft = flag("--speculative-draft-model-path", "")
    if draft:
        snapshot = re.search(r"/models--([^/]+)/snapshots/([^/]+)", draft)
        if snapshot:
            draft, draft_revision = snapshot[1].replace("--", "/", 1), snapshot[2]
        elif draft.startswith(("/", "./", "../", "~")):
            # A local custom drafter is supplied through the recipe's mount.
            draft = ""
        draft_revision = draft_revision or REVISIONS.get(draft, "")
    values = [model, revision, served, str(port), draft, draft_revision]
    if not model or not served:
        raise ValueError("model and served alias must be nonempty")
    if any(not isinstance(v, str) or any(c in v for c in "|\r\n") for v in values):
        raise ValueError("recipe metadata must be single-line strings without '|'")
    return values


if __name__ == "__main__":
    values = metadata(sys.argv[1], sys.argv[2])
    print(render(sys.argv[1]) if sys.argv[3:] == ["--render"] else "|".join(values))
