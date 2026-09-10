#!/usr/bin/env python3
"""Per-prompt TTFT, decode rate and speculative acceptance against a running DeepSeek-V4 server.

    measure.py BASE MODEL [OUT_JSON] [--trials N]

Decode is quoted the way both public deployments quote it: completion tokens divided by the time
from the first streamed token to the last, so time-to-first-token is reported separately and does
not drag a 200-token answer down. Token counts come from the server's usage block, not from
counting stream deltas, because a delta is a decode step and a step carries several tokens under
speculative decoding. Acceptance is a counter delta around each request, so it belongs to the
content that produced it; this drafter accepts a third on prose and three quarters on structured
output, and a pooled number hides that. Warms first: every published figure is warm.
"""
import json, statistics, sys, time, urllib.request

args = [a for a in sys.argv[1:] if not a.startswith("--")]
BASE, MODEL = args[0], args[1]
OUT = args[2] if len(args) > 2 else None
TRIALS = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else 1
THINKING = "--thinking" in sys.argv   # default off: both public deployments quote thinking-off decode

PROMPTS = [
    ("counting", "Count from 1 to 300, one number per line.", 900),
    ("json",     "Emit a JSON array of 60 objects, each with id, name, email and city fields.", 800),
    ("code",     "Write a Python binary search tree with insert, delete and in-order traversal.", 600),
    ("table",    "Write the full 12 by 12 multiplication table as a markdown table.", 900),
    ("prose",    "Write a 300 word narrative about a lighthouse keeper.", 450),
]


def counters():
    d = {"accepted": 0.0, "drafts": 0.0}
    try:
        text = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    except Exception:
        return d
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            d["accepted"] += float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            d["drafts"] += float(line.split()[-1])
    return d


def ask(prompt, max_tokens):
    """Returns (completion_tokens, ttft_s, decode_s)."""
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "chat_template_kwargs": {"thinking": THINKING},
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; t_last = t0; n = 0; finish = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                n = ev["usage"].get("completion_tokens", n)
            ch = ev.get("choices") or []
            if ch:
                delta = ch[0].get("delta") or {}
                finish = ch[0].get("finish_reason") or finish
                # any non-empty string field is a generated token: content, reasoning, reasoning_content
                if any(isinstance(v, str) and v for v in delta.values()):
                    now = time.time()
                    if t_first is None: t_first = now
                    t_last = now
    if t_first is None or t_last - t_first < 0.05:
        raise RuntimeError(f"no streamed tokens observed (finish={finish}, usage tokens={n})")
    return n, t_first - t0, t_last - t_first, finish


print("warming", flush=True)
for _ in range(3):
    ask("Write a 700 word essay about the sea.", 700)

rows = []
for label, prompt, maxtok in PROMPTS:
    dec, ttft, acc = [], [], []
    for _ in range(TRIALS):
        c0 = counters(); n, tf, td, fin = ask(prompt, maxtok); c1 = counters()
        dd = c1["drafts"] - c0["drafts"]; da = c1["accepted"] - c0["accepted"]
        dec.append(n / td); ttft.append(tf); acc.append(da / dd if dd else 0.0)
    row = {"workload": label, "tokens": n, "trials": TRIALS, "finish": fin, "thinking": THINKING,
           "decode_tok_s": round(statistics.median(dec), 1),
           "decode_min": round(min(dec), 1), "decode_max": round(max(dec), 1),
           "ttft_s": round(statistics.median(ttft), 2),
           "accepted_per_cycle": round(statistics.median(acc), 2)}
    rows.append(row)
    print(f"  {label:9s} {n:4d} tok  decode {row['decode_tok_s']:6.1f} tok/s"
          f"  ({row['decode_min']}-{row['decode_max']})  ttft {row['ttft_s']:.2f}s"
          f"  accept {row['accepted_per_cycle']:.2f}  finish={fin}", flush=True)

if rows:
    real = [r["decode_tok_s"] for r in rows if r["workload"] != "counting"]
    print(f"\nreal-prompt median {statistics.median(real):.1f} tok/s (counting excluded, acceptance ceiling only)")
if OUT:
    json.dump(rows, open(OUT, "w"), indent=1); print("wrote", OUT)
