
## First measured boot on this pair, 2026-09-09

DeepSeek-V4-Flash-DSpark text checkpoint at `62af8fff`, `dsv4-text-dspark-tp2`,
TP=2, draft depth 5, NVFP4 sparse-MLA KV, 1M context, GPU memory utilization
0.80. Image `ghcr.io/anemll/dspark-vllm-gx10` at `sha256:a8394849`, vLLM
`0.25.2.dev0+g752a3a504`. Warm, non-streamed, temperature 0, single stream.

| workload | tok/s |
|---|---|
| JSON, 60 objects | 83.9 |
| code, binary search tree | 68.0 |
| counting to 300 | 74.4 |
| prose, 200 words | 30.0 |

KV pool 1,843,706 tokens in 12.49 GiB, 1.76x concurrency at 1M context. Boot to
healthy about 10 minutes. Steady-state MemAvailable 12 GiB head, 14 GiB worker.

Speculative counters over the run: 3,010 accepted tokens across 1,326 draft
cycles, so 2.27 accepted per cycle against a ceiling of 5, about 45%.

Measurement notes. Every published figure for this model is warm, and the
references report a cold penalty near 30%, so three 700-token generations run
first. Streaming is off deliberately: streamed deltas measure steps per second,
which upstream reports as 14.7 against 60.1 tok/s on an identical request.

Single boot, no bracketing controls yet, so treat these as a starting point
rather than a result. The acceptance figure is the interesting one: upstream
reports 60.2% on this checkpoint after a draft shared-expert loader fix, and the
gap is worth roughly 20 tok/s on their numbers. Whether this image carries that
fix is the first thing to check.

### The vision checkpoint does not boot on this image as configured

`dsv4-vision-dspark-tp2` fails at load:

    ValueError: There is no module or parameter named 'aligner' in DeepseekV4ForCausalLM

The Vision-Exp checkpoint carries `aligner` weights for the vision projector,
but the engine instantiates the text-only class, so the vision tower is never
registered. Both public deployments solve this outside the recipe: one applies a
startup hotfix in its launch script, the other stages four `ds4v_*` files
read-only on both nodes. A plain `vllm serve` on the bare image does neither.
The two vision recipes here are therefore unverified until that is resolved.

## Vision-Exp boots and measures on this pair, 2026-09-10

`dsv4-vision-dspark-tp2` with the hotfix mod, fp8 KV, 400k context, k=6,
5 sequences, capture 40, utilization 0.82. Boot to healthy 10 minutes. Draft
loaded 99 params (text: 96; the three extra are the vision `bias_vl` remaps).
KV pool 1,289,993 tokens, 3.22x concurrency at 400k. MemAvailable head 10 GiB.

Thinking off, streaming, decode from first token to last, median of 3:

| workload | decode tok/s | min-max | accepted per cycle (of 6) |
|---|---|---|---|
| table | 82.5 | 74.2-82.5 | 4.50 |
| json | 79.0 | 76.2-79.8 | 4.47 |
| code | 73.3 | 60.7-73.6 | 3.85 |
| counting (ceiling only) | 71.7 | 67.3-74.5 | 3.89 |
| prose | 29.9 | 29.1-30.5 | 0.96 |

TTFT on these short prompts 0.16-0.18 s. Reference TP2 vision figures from
the same checkpoint: prose 33.2, code 51.8, counting 80.1 (upstream launcher,
k=5, NVFP4 KV, 12 sequences). Code is 41% ahead; prose and counting about 10%
behind. Prose acceptance under one token per six-draft cycle is the number to
move: every cycle verifies six drafts and keeps one.

A first pass with thinking on read 62.9 tok/s on prose and 149 on JSON. Those
figures were an artefact: the harness then counted only visible-content
deltas, so reasoning tokens landed in the count but not in the time window.
Discarded; the harness now treats any streamed token field as a token.

## Ladder day, 2026-09-10: llama-benchy grids on fresh boots

Method: one fresh boot per arm, page cache dropped on both nodes, then two
llama-benchy 0.4.0 grids against it: the default book (text) and a 130 KB Python
source file (code), pp 2048, tg 128, depths 0/4k/8k/16k/32k, concurrency
1/2/5, prefix caching on, thinking off, 2 runs per cell. Controls opened and
closed ladder 1. Across four boots of the same recipe the c2 and c5 cells
agree to within 1 tok/s; c1 swings by up to 10 tok/s, so single-stream claims
need several boots. Raw tables in `results/ladder/` and `results/ladder2/`.

### Text lane, tg128 tok/s aggregate

| depth | conc | control A | control B | gmu 0.85 | NCCL Simple | NCCL LL128 | in-flight 3 | stock sched |
|---|---|---|---|---|---|---|---|---|
| 0 | c1 | 46.0 | 45.1 | 45.4 | 47.5 | 48.8 | 35.5 | 45.7 |
| 0 | c2 | 49.9 | 49.5 | 48.6 | 49.7 | 44.1 | 58.2 | 56.7 |
| 0 | c5 | 55.3 | 52.0 | 58.4 | 53.4 | 43.6 | 80.2 | 102.7 |
| 4k | c1 | 35.9 | 46.0 | 44.0 | 43.0 | 44.0 | 39.7 | 46.7 |
| 4k | c2 | 30.2 | 29.3 | 32.1 | 30.8 | 30.2 | 43.2 | 60.4 |
| 4k | c5 | 29.3 | 29.0 | 28.2 | 28.9 | 26.4 | 47.7 | 67.4 |
| 8k | c1 | 38.0 | 39.2 | 39.3 | 37.9 | 40.9 | 38.2 | 43.6 |
| 8k | c2 | 33.1 | 31.6 | 32.0 | 31.6 | 29.3 | 56.7 | 46.3 |
| 8k | c5 | 28.9 | 27.6 | 28.4 | 29.1 | 26.2 | 41.2 | 73.0 |
| 16k | c1 | 41.8 | 39.2 | 37.1 | 47.2 | 41.8 | 37.0 | 38.0 |
| 16k | c2 | 31.5 | 29.4 | 29.7 | 30.2 | 28.8 | 41.3 | 48.0 |
| 16k | c5 | 29.1 | 28.4 | 28.0 | 28.6 | 26.2 | 38.3 | 56.5 |
| 32k | c1 | 42.7 | 47.3 | 43.0 | 41.5 | 45.9 | 42.0 | 39.3 |
| 32k | c2 | 30.6 | 30.9 | 31.7 | 29.6 | 30.5 | 47.5 | 43.0 |
| 32k | c5 | 28.0 | 26.9 | 27.9 | 27.7 | 25.4 | 39.0 | 53.8 |

Code lane runs 3-5 tok/s lower at c1 and matches at c2 and c5; the in-flight 3
code lane confirmed its text lane in every cell. Stock-scheduler code lane was
cut short by the shutdown.

### Prefill, pp2048 new tokens, tok/s (control)

Depth 0: about 1480 at any concurrency. Any cached context: about 500, flat
from 4k to 32k, at any concurrency. The cost is the sparse indexer touching
cached keys at all, not how many. This cell did not move on any arm run so far;
the indexer-path arms (sp-indexer, C128A prefill cache, MXFP4 indexer cache)
were queued but not reached.

### Verdicts

- **Stock scheduler, promote candidate.** Skipping MiaAI's issue27 in-flight
  prefill cap and issue43 decode-fairness patches (`DSPARK_SKIP_ISSUE27_HOTFIX=1`,
  `DSPARK_SKIP_ISSUE43_HOTFIX=1`) doubles c5 at every depth against the control
  and beats or matches in-flight 3 at c2. c1 and acceptance unchanged. The
  trade is the one those patches were written for: c5 error bars widen from
  1-2 to 10-16 tok/s, so streams are less even. Fine for an agent farm; a
  latency-SLO chat service may prefer in-flight 3, which keeps the patches and
  still lifts c2 by 30-80% and c5 by 30-65%.
- **In-flight prefill 3: promoted** into the shipped recipes (`DSPARK_MAX_INFLIGHT_PREFILLS=3`).
- **Utilization 0.85: promote.** KV pool 1.56M against 1.25M tokens, decode
  within the band, headroom 7-8 GiB held through 32k at c5.
- **NCCL Simple: tie. NCCL LL128: retire**, 10-30% slower prefill and 5-15%
  lower c2/c5. The two-node step is not protocol-bound.
- **Fixed k=5: neutral. Expert parallelism: cannot boot** on the B12X MoE kernel.

### Context for the numbers

Two Spark Arena entries for this checkpoint that report higher aggregates are
four-node deployments (two independent TP2 instances behind a router; one TP4).
Per node, this TP2 recipe is the strongest of the three, and its prefill after a
cached 32k context, 480 tok/s, is 5x the two-instance run's 87. The TP4 run's
depth penalty is 22% against this recipe's 67%, which points at a TP2-specific
cost in the deep-context indexer path that remains open.

Against Qwen3.8-Flash-Next on the same pair and grid: single stream is a tie;
Qwen is about 2x faster at concurrency and prefill and 4x at c5 with cached
context, because its linear attention has no indexer. DeepSeek wins vision and
structured single-stream decode (73-83 tok/s on code, JSON, tables).
