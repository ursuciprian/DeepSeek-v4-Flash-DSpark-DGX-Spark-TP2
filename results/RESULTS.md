
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
