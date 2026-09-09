# DeepSeek-V4-Flash with DSpark on two DGX Sparks

Serving recipes for `DeepSeek-V4-Flash` at tensor parallel 2 across two NVIDIA
DGX Spark (GB10) over ConnectX-7, with DSpark speculative decoding, for
[sparkrun](https://sparkrun.dev). Three recipes: vision with an fp8 KV cache,
vision with the NVFP4 sparse-MLA cache and a 1M window, and the text checkpoint.

`sparkrun registry add` and run. The container applies its own hotfixes at
startup, so there is nothing to bind-mount and no absolute path to edit.

## Which recipe

| You are serving | Recipe | Context | KV cache | Draft depth |
|---|---|---|---|---|
| Images and text, several conversations | `dsv4-vision-dspark-tp2` | 400k | fp8 | 6 |
| Long documents kept resident, images | `dsv4-vision-dspark-tp2-nvfp4kv` | 1M | NVFP4 sparse-MLA | 6 |
| Text only, largest KV pool | `dsv4-text-dspark-tp2` | 1M | NVFP4 sparse-MLA | 5 |

```sh
sparkrun registry add https://github.com/ursuciprian/DeepSeek-v4-Flash-DSpark-DGX-Spark-TP2
sparkrun run @dsv4-dspark/dsv4-vision-dspark-tp2 --cluster <your-cluster> --tp 2
```

Or clone and use the launcher, which checks both nodes, mirrors the checkpoint
over the fast link and drops the page cache first:

```sh
scripts/run.sh vision --check      # validate, launch nothing
scripts/run.sh vision
scripts/run.sh vision-1m
scripts/run.sh text
```

The checkpoint is about 157 GB and the image about 9 GB, so the first launch is
long. `scripts/validate_recipes.py` checks a recipe before you spend that time.

## Before the first launch

**Fabric.** The recipes pin NCCL to this cluster's interface and HCAs. Print
and patch yours:

```sh
WORKER_IP=<worker ip on the fast link> scripts/detect-fabric.sh --write
```

Left wrong, NCCL falls back to TCP over the management link and decode jitter
rises sharply.

**earlyoom must be off on both nodes.** Under deep-context load it will kill
the engine. This model runs close to the memory ceiling by design.

**The snapshot must be complete on both nodes.** `--tokenizer-mode deepseek_v4`
loads the checkpoint's own encoder from `encoding/encoding_dsv4.py`. A filtered
or partial download serves garbled text instead of failing, which is a slow way
to lose an evening. `scripts/run.sh` checks for it on both nodes before
launching. Neither this repository nor the two references ship a chat template;
the encoder plus `--default-chat-template-kwargs` is the whole path, which is
why the official checkpoint has no `chat_template` in `tokenizer_config.json`.

**Drop the page cache on both nodes.** On unified memory a warm cache starves
the GPU allocator part-way through the load, and `free -g` misreports it.

```sh
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

## What these recipes fix

They started from a working TP2 configuration and from the two public recipes
credited below. Each change below is a defect or a documented cost, not a
preference.

**CUDA graph capture was too small.** The rule is
`max_num_seqs x (draft depth + 1)`, rounded up to a multiple of 8. At five
sequences and depth 6 that is 40, not 24. Below it, every decode batch larger
than the captured size runs eager; upstream measured about 12% lost at six
streams from a capture that truncated by a single step. `dsv4-vision-dspark-tp2`
uses 40, the 1M variants 48.

**Prefix caching had no retention interval.** Over sparse MLA the cache keeps
state on its own grid, and a reused prefix that lands between kept positions
finds nothing to resume from, so the turn re-prefills in full. All three
recipes set `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`. If you have watched
this happen on another hybrid model, it is the same failure.

**A stale env var contradicted the launch flag.** A `SPECULATIVE_CONFIG`
environment variable asked for greedy drafting while the command line asked for
probabilistic. Greedy is the documented cause of garbled output on the first
requests after a boot. The variable is gone; the flag is what the engine reads.

**JIT caches pointed into the model cache.** DeepGEMM, FlashInfer, Triton,
Inductor and torch-extension caches must be node-local. In the model cache
directory, which is often shared or NFS-mounted, every rank recompiles and the
boot can deadlock. Each now has its own directory under a node-local root.

**Draft depth is checkpoint-specific and easy to get wrong.** Vision-Exp sets
`n_predict=3`, so the depth must be at least 5 and a multiple of 3: 6 works, 5
is rejected. The text checkpoint sets `n_predict=5`, so it takes 5. Copying a
depth between them fails at load with a tensor size mismatch, and the capture
size has to move with it.

**The image was a local tag.** It is now `ghcr.io/anemll/dspark-vllm-gx10`
pinned by digest, so anyone can pull exactly what was measured and a moved tag
cannot change the build underneath.

**One prefill in flight.** `DSPARK_MAX_INFLIGHT_PREFILLS=1` keeps decode fair;
raising it to 2 widens the time-to-first-token spread across concurrent
requests without improving the median.

## Reasoning, tools and images

Thinking is on with `reasoning_effort: low`. Two consequences worth knowing:

- `max_tokens` covers the reasoning, the visible answer and any tool markup
  together. Small caps return `finish_reason: length` with empty content. Budget
  in tens of thousands of tokens if you raise the effort.
- A client stop string can fire inside the reasoning block and decapitate the
  answer. The image suppresses stops until the reasoning ends; leave that on.

Images go in `user` messages only. A structured image part on a `system`,
`assistant` or `tool` message returns HTTP 400, and because most clients do not
retry, the rejected turn stays in the history and every later message fails too.
Eight images per request.

## Validating a change

```sh
scripts/validate_recipes.py recipes
```

Beyond the schema, it checks the capture-size rule against the draft depth and
sequence count, greedy drafting, prefix caching without a retention interval,
JIT caches inside the model cache, contradictory speculative settings, moving
image tags, unset fabric and credentials in `env`.

## Hardware

- 2x DGX Spark: GB10, 128 GB LPDDR5X unified, about 273 GB/s per node.
- ConnectX-7 RoCE between the nodes, GID index 3.
- About 157 GB of disk for the checkpoint per node, plus the 9 GB image.

## Status

The recipes are written and validated; measurements on this pair are not in yet,
so no throughput figures are published here. `results/` will carry the grids and
the method when they are.

## Credits

MiaAI-Lab's [DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
for the two-node DSpark recipe, the hotfix set and the serve-shape knobs.
tonyd2wild's [DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
for the NVFP4 sparse-MLA cache path, the draft-depth and capture-size analysis,
and the acceptance measurements. hyudryu's SparkDeck for the deployment
orchestration this grew out of. The packaging, the corrections above and the
validator are mine, and so are any mistakes.

Apache-2.0.
