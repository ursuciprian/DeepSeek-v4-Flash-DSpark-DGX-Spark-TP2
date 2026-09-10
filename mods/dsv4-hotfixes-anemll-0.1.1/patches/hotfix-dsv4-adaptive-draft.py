#!/usr/bin/env python3
"""Adaptive DSpark draft length for anemll dspark-vllm-gx10 0.1.1 (vLLM 0.25.2.dev0+g752a3a504).

Why. The DSpark drafter emits a fixed block of ``num_speculative_tokens`` drafts every step. On
prose it keeps about one of six (measured 0.93-1.01 accepted per cycle on this pair, and a third
party reports 19-22% on the same weights, MiaAI-Lab issue #260), yet the target still verifies
all six. Verification on a MoE is memory-bound and every extra verified token activates more
experts, so a six-token block the target will reject after position one costs real step time.

What. The sequential Markov sampling loop already produces the draft distribution at each
position. This patch records the top-1 probability of that distribution as a per-position
confidence, cuts each request's draft at the first position whose confidence falls below a
threshold, and writes -1 into the cut positions. The runner-side handler then always copies
the draft block to the host and strips the -1 tail, so the scheduler counts only the kept
drafts, the runner verifies only those, and the rejection sampler never sees the rest. Every
op added inside the sampling loop is a fixed-shape tensor op, so FULL CUDA-graph capture of
the draft step is unaffected.

Env (read at speculator construction):
  DSPARK_DRAFT_CONF_THRESHOLD  float in [0,1]; 0 (default) = patch inert, stock behaviour
  DSPARK_DRAFT_MIN_LEN         int >= 0, default 1; never cut below this many drafts

Cost. One extra device-to-host copy of num_reqs x k int32 per step, on the same async path the
structured-output case already uses. Measured, not assumed: run the arm with the threshold at 0
and at the candidate value on fresh boots.

Source-exact and fail-closed: anchors must match exactly once or nothing is written.
"""
from __future__ import annotations

import sys
from pathlib import Path

V = Path("/usr/local/lib/python3.12/dist-packages/vllm")
SPEC = V / "v1/worker/gpu/spec_decode/dspark/speculator.py"
UTIL = V / "v1/worker/gpu/spec_decode/utils.py"
MARK = "# [adaptive-draft]"


def patch(path: Path, edits: list[tuple[str, str]]) -> str:
    src = path.read_text()
    if MARK in src:
        return "already applied"
    out = src
    for anchor, replacement in edits:
        n = out.count(anchor)
        if n != 1:
            sys.exit(f"FATAL: {path.name}: anchor found {n} times, expected 1:\n{anchor[:160]}")
        out = out.replace(anchor, replacement)
    compile(out, str(path), "exec")
    path.write_text(out)
    return "applied"


# --- speculator: confidence + truncation -------------------------------------------------------
spec_edits = [
    (
        "from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model\n",
        "from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model\n"
        "import os  " + MARK + "\n",
    ),
    (
        "        # Reduced-vocab probabilistic drafting only; set in load_draft_model.\n"
        "        self._d2t_scatter_index: torch.Tensor | None = None\n"
        "        self._draft_scatter_buf: torch.Tensor | None = None\n",
        "        # Reduced-vocab probabilistic drafting only; set in load_draft_model.\n"
        "        self._d2t_scatter_index: torch.Tensor | None = None\n"
        "        self._draft_scatter_buf: torch.Tensor | None = None\n"
        "\n"
        "        " + MARK + " per-position top-1 confidence and threshold truncation\n"
        "        self._conf_threshold = float(os.getenv(\"DSPARK_DRAFT_CONF_THRESHOLD\", \"0\"))\n"
        "        if not 0.0 <= self._conf_threshold <= 1.0:\n"
        "            raise ValueError(\"DSPARK_DRAFT_CONF_THRESHOLD must be in [0, 1]\")\n"
        "        self._conf_min_len = max(0, int(os.getenv(\"DSPARK_DRAFT_MIN_LEN\", \"1\")))\n"
        "        self._conf = torch.zeros(\n"
        "            self.max_num_reqs, self.num_speculative_steps, dtype=torch.float32, device=device\n"
        "        )\n"
        "        self._pos_cols = torch.arange(\n"
        "            self.num_speculative_steps, dtype=torch.int32, device=device\n"
        "        ).unsqueeze(0)\n"
        "        if self._conf_threshold > 0.0:\n"
        "            from vllm.logger import init_logger\n"
        "            init_logger(__name__).info(\n"
        "                \"DSpark adaptive draft: threshold %.3f, min_len %d\",\n"
        "                self._conf_threshold, self._conf_min_len,\n"
        "            )\n",
    ),
    (
        "            logits_i = base_logits[:, i] + bias\n"
        "            if self.draft_logits is not None:\n",
        "            logits_i = base_logits[:, i] + bias\n"
        "            if self._conf_threshold > 0.0:  " + MARK + "\n"
        "                self._conf[:num_reqs, i] = torch.softmax(\n"
        "                    logits_i.float(), dim=-1\n"
        "                ).amax(dim=-1)\n"
        "            if self.draft_logits is not None:\n",
    ),
    (
        "            self.draft_tokens[:num_reqs, i] = draft_sampled_i\n"
        "            prev = draft_sampled_i\n",
        "            self.draft_tokens[:num_reqs, i] = draft_sampled_i\n"
        "            prev = draft_sampled_i\n"
        "        if self._conf_threshold > 0.0:  " + MARK + "\n"
        "            # keep the prefix up to the first low-confidence position, never below min_len\n"
        "            ok = (self._conf[:num_reqs] >= self._conf_threshold).to(torch.int32)\n"
        "            keep = ok.cumprod(dim=1).sum(dim=1).clamp_(min=self._conf_min_len)\n"
        "            cut = self._pos_cols >= keep.unsqueeze(1)\n"
        "            self.draft_tokens[:num_reqs].masked_fill_(cut, -1)\n",
    ),
]

# --- handler: always copy the block to host, strip the -1 tail per request ----------------------
util_edits = [
    (
        "        self.req_ids = input_batch.req_ids\n"
        "        self.num_draft_tokens = draft_tokens.shape[1]\n"
        "        if not input_batch.has_structured_output_reqs:\n",
        "        self.req_ids = input_batch.req_ids\n"
        "        self.num_draft_tokens = draft_tokens.shape[1]\n"
        "        _adaptive = float(__import__(\"os\").getenv(\"DSPARK_DRAFT_CONF_THRESHOLD\", \"0\")) > 0  " + MARK + "\n"
        "        if not input_batch.has_structured_output_reqs and not _adaptive:\n",
    ),
    (
        "            self.copy_event.synchronize()\n"
        "            draft_token_ids = self.draft_tokens_np.tolist()\n",
        "            self.copy_event.synchronize()\n"
        "            draft_token_ids = self.draft_tokens_np.tolist()\n"
        "            " + MARK + " -1 marks a cut draft slot; drop from the first one so the scheduler\n"
        "            # and the runner count only the kept drafts for this request\n"
        "            for _r, _row in enumerate(draft_token_ids):\n"
        "                if -1 in _row:\n"
        "                    draft_token_ids[_r] = _row[: _row.index(-1)]\n",
    ),
]

if __name__ == "__main__":
    if "--check" in sys.argv:
        for p, edits in ((SPEC, spec_edits), (UTIL, util_edits)):
            s = p.read_text()
            state = "applied" if MARK in s else ("ready" if all(s.count(a) == 1 for a, _ in edits) else "DRIFT")
            print(f"adaptive-draft {p.name}: {state}")
        sys.exit(0)
    r1 = patch(SPEC, spec_edits)
    r2 = patch(UTIL, util_edits)
    print(f"[adaptive-draft] speculator.py: {r1}; utils.py: {r2}")
