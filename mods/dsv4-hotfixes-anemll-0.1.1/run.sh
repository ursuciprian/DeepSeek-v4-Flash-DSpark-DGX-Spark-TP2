#!/usr/bin/env bash
# Startup hotfix train for ghcr.io/anemll/dspark-vllm-gx10 0.1.1 (vLLM 0.25.2.dev0+g752a3a504).
#
# The image's DeepseekV4ForCausalLM is text-only, so the Vision-Exp checkpoint fails at load with
# "There is no module or parameter named 'aligner'". These patchers, written by MiaAI-Lab for the
# same image and applied here in the same order as their compose entrypoint, add the vision tower,
# the checkpoint's own encoder, and a set of correctness and scheduling fixes. Every patcher is
# source-exact and fail-closed: if the image underneath has changed, the boot stops here instead
# of serving something subtly wrong.
#
# Always applied: encoder copy + reasoning-effort mapping, issue21, issue55, issue22 (nvfp4 MLA),
# gb10 spin-wait, issue117, the six shell hotfixes, vision-exp, empty-encoder-output, issue27,
# issue43, issue26, issue133, runtime-ablation, suppress-stops-in-reasoning.
# Opt-in through env, default off, same names as upstream: DSPARK_ENABLE_ISSUE31_GPU_HOTFIX,
# DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK, DSPARK_ENABLE_SP_INDEXER, DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS,
# DSPARK_ENABLE_ADAPTIVE_CHUNK, DSPARK_ENABLE_REPLICATE_MARKOV, DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX,
# DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN, DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX,
# DSPARK_ENABLE_ISSUE191_TOOLCALL_FAILCLOSED, DSPARK_ENABLE_DSPARK_BLOCK_K, DSPARK_ENABLE_ROPE_SWA_FIX,
# DSPARK_ENABLE_DSPARK_SWA_PREFIX, DSPARK_ENABLE_DSML_RECOVERY, DSPARK_ENABLE_MXFP4_INDEXER_CACHE,
# DSPARK_ENABLE_C128A_PREFILL_CACHE. Skips: DSPARK_SKIP_ISSUE22_HOTFIX, DSPARK_SKIP_SPIN_WAIT_HOTFIX,
# DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX, DSPARK_SKIP_HOTFIX, DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V=/usr/local/lib/python3.12/dist-packages/vllm
[ -d "$V/models/deepseek_v4/nvidia" ] || { echo "mod dsv4-hotfixes: $V/models/deepseek_v4 not in this image, skipping"; exit 0; }

# Lay the files out where the patchers expect them (paths are baked into the scripts).
mkdir -p /opt/dspark-patches
cp "$HERE"/patches/hotfix-*.py /opt/
cp "$HERE"/patches/hotfix-*.sh /opt/dspark-patches/
cp "$HERE"/patches/hotfix-vllm-issue117-shm-ring-buffer.py /opt/dspark-patches/
rm -rf /opt/dspark-patches/vision_exp; cp -R "$HERE/patches/vision_exp" /opt/dspark-patches/vision_exp
export PATH="/usr/local/cuda/bin:/usr/local/bin:${PATH:-}"

on() { [ "${!1:-0}" = "1" ]; }
run_py() { echo "  hotfix: $1"; python3 "/opt/$1"; }
run_sh() { echo "  hotfix: $1"; bash "/opt/dspark-patches/$1"; }

# 1. The checkpoint's own encoder replaces the image's copy; the effort mapping makes anything
#    that is not max/xhigh/high resolve to low instead of high.
# Prefer the served checkpoint's encoder (DSPARK_MODEL, set by the recipe), then Vision-Exp, then
# any V4-Flash snapshot. Two checkpoints can share the cache and the text one's encoder has no
# image placeholder, which the vision patcher rejects as drift.
ENC=""
HUB=/cache/huggingface/hub
for m in "${DSPARK_MODEL:-}" deepseek-ai/DeepSeek-V4-Flash-Vision-Exp deepseek-ai/DeepSeek-V4-Flash-DSpark deepseek-ai/DeepSeek-V4-Flash-0731; do
  [ -n "$m" ] || continue
  for c in "$HUB/models--${m//\//--}"/snapshots/*/encoding/encoding_dsv4.py; do
    [ -f "$c" ] && { ENC="$c"; break 2; }
  done
done
if [ -n "$ENC" ]; then
  echo "  encoder: $ENC"
  cp "$ENC" "$V/tokenizers/deepseek_v4_encoding.py"
  python3 - "$V/tokenizers/deepseek_v4.py" <<'PY'
import sys; from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()
old = ('elif reasoning_effort in ("max", "xhigh"):\n                reasoning_effort = "max"\n'
       '            else:\n                reasoning_effort = "high"')
new = ('elif reasoning_effort in ("max", "xhigh"):\n                reasoning_effort = "max"\n'
       '            elif reasoning_effort == "high":\n                reasoning_effort = "high"\n'
       '            else:\n                reasoning_effort = "low"')
if new not in s:
    assert old in s, "reasoning-effort anchor missing in deepseek_v4.py"
    p.write_text(s.replace(old, new))
PY
  run_py hotfix-encoding-dsv4-issue21.py
else
  echo "  WARN: encoding_dsv4.py not found in /cache/huggingface; encoder copy and issue21 skipped" >&2
fi

on DSPARK_ENABLE_ISSUE31_GPU_HOTFIX && run_py hotfix-dsv4-issue31-v2-thinking-budget-gpu.py
run_py hotfix-dsv4-issue55-tool-truncation.py
on DSPARK_SKIP_ISSUE22_HOTFIX      || run_sh hotfix-nvfp4-ds-mla-issue22.sh
on DSPARK_SKIP_SPIN_WAIT_HOTFIX    || run_sh hotfix-gb10-spin-wait.sh
if ! on DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX; then
  echo "  hotfix: hotfix-vllm-issue117-shm-ring-buffer.py"
  python3 /opt/dspark-patches/hotfix-vllm-issue117-shm-ring-buffer.py
  python3 /opt/dspark-patches/hotfix-vllm-issue117-shm-ring-buffer.py --status
fi
if ! on DSPARK_SKIP_HOTFIX; then
  for f in hotfix-dsv4-mtp-buffer-50312.sh hotfix-dsv4-skip-topk-49486.sh hotfix-dsv4-dense-prefill-indexer-48407.sh \
           hotfix-dsv4-skip-empty-c128-48957.sh hotfix-dsv4-flashmla-workspace-50298.sh hotfix-dsv4-grammar-advance.sh; do
    run_sh "$f"
  done
fi

# 2. Vision-Exp support and the fixes that sit around it.
run_py hotfix-dsv4-vision-exp.py
on DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK && run_py hotfix-dsv4-issue141-sparse-mla-decode-chunk.py
on DSPARK_ENABLE_SP_INDEXER               && run_py hotfix-dsv4-sp-indexer-prefill.py
on DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS      && run_sh hotfix-deepgemm-sm121-mqa-header-alias.sh
run_py hotfix-vllm-empty-encoder-output.py
run_py hotfix-dsv4-issue27-partial-prefill-concurrency.py
on DSPARK_ENABLE_ADAPTIVE_CHUNK            && run_py hotfix-dsv4-adaptive-prefill-chunk.py
on DSPARK_ENABLE_REPLICATE_MARKOV          && run_py hotfix-dsv4-replicate-markov-head.py
run_py hotfix-dsv4-issue43-decode-fairness-and-diag.py
run_py hotfix-dsv4-issue26-hybrid-swa-min.py
run_py hotfix-dsv4-issue133-triton-specialization.py
run_py hotfix-dsv4-runtime-ablation.py
on DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX       || run_py hotfix-dsv4-suppress-stops-in-reasoning.py
on DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX    && run_py hotfix-dsv4-assistant-final-continuation.py
on DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN     && run_py hotfix-dsv4-issue144-effort-align.py
on DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX  && run_py hotfix-vllm-issue136-xgrammar-termination.py
on DSPARK_ENABLE_ISSUE191_TOOLCALL_FAILCLOSED && run_py hotfix-vllm-issue191-toolcall-failclosed.py
on DSPARK_ENABLE_DSPARK_BLOCK_K            && run_py hotfix-vllm-dspark-block-k.py
on DSPARK_ENABLE_ROPE_SWA_FIX              && run_py hotfix-vllm-rope-swa-fix.py
on DSPARK_ENABLE_DSPARK_SWA_PREFIX         && run_py hotfix-vllm-dspark-swa-prefix.py
on DSPARK_ENABLE_DSML_RECOVERY             && run_py hotfix-vllm-dsml-recovery.py
on DSPARK_ENABLE_MXFP4_INDEXER_CACHE       && run_py hotfix-vllm-mxfp4-indexer-cache.py
on DSPARK_ENABLE_C128A_PREFILL_CACHE       && run_py hotfix-vllm-c128a-prefill-cache.py
echo "mod dsv4-hotfixes: applied"
