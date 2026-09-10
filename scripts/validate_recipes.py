#!/usr/bin/env python3
"""Recipe validator for this project's Qwen3.8-Flash-Next recipes.

  scripts/validate_recipes.py [PATH ...]        # default: recipes/ under the repo root

Complements `sparkrun recipe validate`, which checks the schema. Every rule here comes from a
failure this project actually hit, so a clean run means the recipe will not fail in one of the
ways we already paid for. Exit code 1 if any ERROR, 0 if only WARN/INFO.
"""
from __future__ import annotations
import re, sys, subprocess
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml required")

REPO = Path(__file__).resolve().parents[1]
ERRORS: list[str] = []
WARNS: list[str] = []

# in-container path -> what our overlays are for; two mounts on one path is always a mistake
SGLANG_QSA_TARGET = "qwen_sparse_attn_backend.py"
# sglang builds from 2026-09-03 on carry upstream's SM121 kernel (#36845); mounting the older
# guard extension over it reinstates the path that corrupts >~95k context on SM121 (#36806)
SEP03_DIGEST = "5ae5816783d58e2e"


def err(f: Path, msg: str) -> None: ERRORS.append(f"{f.name}: ERROR {msg}")
def warn(f: Path, msg: str) -> None: WARNS.append(f"{f.name}: WARN  {msg}")


def check(path: Path) -> None:
    raw = path.read_text()
    body = "\n".join(l for l in raw.splitlines() if not l.startswith("#"))
    try:
        r = yaml.safe_load(body)
    except Exception as e:
        return err(path, f"YAML does not parse: {e}")
    if not isinstance(r, dict):
        return err(path, "top level is not a mapping")

    cmd_raw = r.get("command", "") or ""
    # substitute {placeholders} from defaults so the flag checks see what the engine will get
    cmd = cmd_raw
    for _ in range(3):
        for _k, _v in (r.get("defaults") or {}).items():
            cmd = cmd.replace("{" + str(_k) + "}", str(_v))
    env = r.get("env", {}) or {}
    ex = r.get("executor_config", {}) or {}
    vols = ex.get("volumes", []) or []
    mods = r.get("mods", []) or []
    container = r.get("container", "")
    runtime = (r.get("runtime") or "").lower()

    # --- schema limits the tooling actually enforces
    for k in ("model", "container"):
        if not r.get(k): err(path, f"missing required field `{k}`")
    rv = str(r.get("recipe_version", "1"))
    if rv.isdigit() and int(rv) > 2:
        err(path, f"recipe_version {rv}: sparkrun accepts at most 2, tag the revision in `name`/`metadata` instead")

    # --- image identity: a moving or local-only tag is the fastest way to make a recipe
    # unreproducible. `qwen38flashnext-sep03` existed only on our nodes (2026-09-09).
    if "@sha256:" not in container:
        if re.search(r":(latest|nightly|main|dev)$", container):
            err(path, f"container `{container}` is a moving tag; pin `repo@sha256:...`")
        else:
            warn(path, f"container `{container}` is a tag, not a digest; it can move or be local-only")

    # --- placeholders in the command must resolve from `defaults` plus {model}
    defaults = r.get("defaults", {}) or {}
    known = set(defaults) | {"model"}
    for ph in set(re.findall(r"(?<!\{)\{([a-z_][a-z0-9_]*)\}", cmd_raw)):
        if ph not in known: err(path, f"command uses {{{ph}}} with no `defaults` entry")

    # --- topology consistency: TP=2 recipes that do not pin both nodes hang at rendezvous
    tp = str(defaults.get("tensor_parallel", "")) or (re.search(r"--tp(?:-size)?\s+(\d+)", cmd) or ["", ""])[1]
    if tp == "2" and (r.get("min_nodes"), r.get("max_nodes")) != (2, 2):
        warn(path, "tensor parallel 2 but min_nodes/max_nodes are not both 2")

    # --- vLLM images declare ENTRYPOINT ["vllm","serve"], which eats the launcher's command
    if "vllm-openai" in container and ex.get("entrypoint", None) != "":
        err(path, "vLLM image needs `executor_config: entrypoint: ''` or the serve command is swallowed")

    # --- fabric: unset NCCL interface silently falls back to TCP over the management link
    if (r.get("max_nodes") or 1) > 1 and not env.get("NCCL_SOCKET_IFNAME"):
        warn(path, "multi-node without NCCL_SOCKET_IFNAME: collectives may fall back to the slow link")

    # --- overlays: two sources on one in-container path, or a source that is not in this repo
    dests: dict[str, str] = {}
    for v in vols:
        parts = str(v).split(":")
        if len(parts) < 2: err(path, f"malformed volume `{v}`"); continue
        src, dst = parts[0], parts[1]
        if dst in dests: err(path, f"two mounts target {dst}")
        dests[dst] = src
        p = Path(src)
        prefix = f"/home/nvidia/GEN-AI/{REPO.name}/"
        if str(p).startswith(prefix):
            if not (REPO / str(p)[len(prefix):]).exists(): err(path, f"mount source does not exist: {src}")
        elif str(p).startswith("/home/nvidia/GEN-AI/"):
            # a path under a sibling checkout: only this machine can satisfy it
            warn(path, f"mount source points at another checkout, not portable: {src}")
        elif not p.exists():
            err(path, f"mount source does not exist: {src}")

    # --- the SM121 correctness trap: the guard extension must not be mounted on a build that
    # already carries upstream's kernel
    guard_via_mod = any("qsa-guard" in str(m) for m in mods)
    sep03_or_newer = SEP03_DIGEST in container or container.endswith(":qwen38flashnext") or "sep03" in container
    if (any(SGLANG_QSA_TARGET in d for d in dests) or guard_via_mod) and sep03_or_newer:
        err(path, "applies the old QSA guard over the 2026-09-03 or newer build, reinstating the kernel "
                  "that corrupts long context on SM121 (sglang #36806). Note the mutable tag "
                  "lmsysorg/sglang:qwen38flashnext moved to that build on 2026-09-03.")

    # --- mods must exist and be runnable
    for m in mods:
        if str(m).startswith("@"):
            # registry-scoped reference (@registry/mod): resolvable only on a host that has that
            # registry added, so it cannot be checked from here
            warn(path, f"mod `{m}` is registry-scoped; the user must `sparkrun registry add` first")
            continue
        rel = str(m).removeprefix("mods/")
        cands = [path.parent / "mods" / rel, REPO / "mods" / rel, path.parent / rel]
        hit = next((c for c in cands if (c / "run.sh").is_file()), None)
        if hit is None:
            err(path, f"mod `{m}` not found (looked for run.sh in {', '.join(str(c) for c in cands)})")
            continue
        rs = (hit / "run.sh").read_text()
        for ref in re.findall(r'\$HERE/([A-Za-z0-9_.\-]+)', rs):
            if not (hit / ref).exists(): err(path, f"mod `{m}` run.sh copies {ref}, which is missing")

    # --- DSpark: draft depth, capture size and sampling are coupled and each has bitten
    if "dspark" in cmd:
        d = r.get("defaults", {})
        k = int(d.get("num_speculative_tokens", 0) or 0)
        seqs = int(d.get("max_num_seqs", 0) or 0)
        cap = int(d.get("max_cudagraph_capture_size", 0) or 0)
        if k and seqs and cap:
            need = -(-(seqs * (k + 1)) // 8) * 8      # round up to a multiple of 8
            if cap < need:
                err(path, f"--max-cudagraph-capture-size {cap} is below max_num_seqs x (k+1) "
                          f"rounded to a multiple of 8 ({need}); decode batches above the captured "
                          "size run eager")
        if '"draft_sample_method":"greedy"' in cmd.replace(" ", ""):
            err(path, "DSpark greedy drafting is the documented cause of garbled output on the "
                      "first requests after a boot; use probabilistic")
        if "--enable-prefix-caching" in cmd and not env.get("VLLM_PREFIX_CACHE_RETENTION_INTERVAL"):
            warn(path, "prefix caching over sparse MLA without VLLM_PREFIX_CACHE_RETENTION_INTERVAL: "
                       "a reused prefix may find no state to resume from")
        for k_env, v in env.items():
            if "SPECULATIVE_CONFIG" in k_env and '"greedy"' in str(v):
                err(path, f"env `{k_env}` contradicts the --speculative-config flag (greedy vs "
                          "probabilistic); remove it, the flag is what the engine reads")
        root = env.get("VLLM_CACHE_ROOT", "")
        if root and ("huggingface" in root or "/cache/hf" in root):
            warn(path, f"VLLM_CACHE_ROOT={root} sits in the model cache, which is often shared or "
                       "NFS-mounted; JIT caches must be node-local")

    # --- flags this stack rejects, learned the expensive way
    if "use_local_argmax_reduction" in cmd and '"use_local_argmax_reduction":true' in cmd.replace(" ", ""):
        err(path, "use_local_argmax_reduction: Qwen4ExpMTP has no get_top_tokens() on vLLM nightly "
                  "8a728663; the engine refuses to start")
    m = re.search(r"--speculative-num-draft-tokens\s+(\d+)", cmd)
    if runtime == "sglang" and m and int(m.group(1)) > 4:
        err(path, f"--speculative-num-draft-tokens {m.group(1)}: the QSA verification block caps at 4")
    if "--allow-auto-truncate" in cmd:
        warn(path, "--allow-auto-truncate silently truncates over-length requests instead of failing")
    if "--mamba-cache-mode all" in cmd:
        warn(path, "mamba-cache-mode all is deprecated and is coerced to `align` on current builds")
    if "--enable-prefix-caching" in cmd and "mtp" in cmd and "disable_eagle_block_drop" not in cmd \
       and "--prefix-cache-retention-interval" not in cmd:
        warn(path, "MTP + prefix caching without disable_eagle_block_drop or a retention interval: "
                   "a prompt's first pass is not reusable before vLLM #53945")

    # --- secrets
    for k, v in env.items():
        if re.search(r"token|secret|password|api_key", k, re.I) and str(v) not in ("", "0", "1"):
            err(path, f"env `{k}` looks like a credential; keep it out of the recipe")


def main() -> int:
    args = sys.argv[1:] or [str(REPO / "recipes")]
    files: list[Path] = []
    for a in args:
        p = Path(a)
        files += sorted(p.rglob("*.yaml")) if p.is_dir() else [p]
    files = [f for f in files if f.is_file()]
    for f in files: check(f)
    for line in WARNS: print(line)
    for line in ERRORS: print(line)
    print(f"\n{len(files)} recipes: {len(ERRORS)} errors, {len(WARNS)} warnings")
    return 1 if ERRORS else 0


if __name__ == "__main__":
    sys.exit(main())
