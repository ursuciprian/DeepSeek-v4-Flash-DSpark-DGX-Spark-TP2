#!/usr/bin/env python3
"""Per-prompt throughput and speculative acceptance against a running DeepSeek-V4 server.

    dsv4-measure.py BASE MODEL [OUT_JSON]

Acceptance is read as a counter delta around each request, so a figure belongs to the content
that produced it. A cumulative read mixes warm-up and every prompt type together, and this
drafter's acceptance swings from about a third on prose to three quarters on structured output,
so the cumulative number is meaningless for comparison.

Warms first: every published figure for this model is warm and the cold penalty is near 30%.
Non-streamed: streamed deltas measure steps per second, not tokens per second.
"""
import json, sys, time, urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None

PROMPTS = [
    ("counting",   "Count from 1 to 300, one number per line.", 900),
    ("json",       "Emit a JSON array of 60 objects, each with id, name, email and city fields.", 800),
    ("code",       "Write a Python binary search tree with insert, delete and in-order traversal.", 600),
    ("table",      "Write the full 12 by 12 multiplication table as a markdown table.", 900),
    ("prose",      "Write a 200 word narrative about a lighthouse keeper.", 300),
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
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": False}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=900))
    el = time.time() - t
    return r.get("usage", {}).get("completion_tokens", 0), el


print("warming", flush=True)
for _ in range(3):
    ask("Write a 700 word essay about the sea.", 700)

rows = []
for label, prompt, maxtok in PROMPTS:
    c0 = counters()
    n, el = ask(prompt, maxtok)
    c1 = counters()
    dd = c1["drafts"] - c0["drafts"]
    da = c1["accepted"] - c0["accepted"]
    per_cycle = da / dd if dd else 0.0
    row = {"workload": label, "tokens": n, "seconds": round(el, 2),
           "tok_s": round(n / el, 1) if el else 0.0,
           "accepted_per_cycle": round(per_cycle, 2), "draft_cycles": int(dd)}
    rows.append(row)
    print(f"  {label:9s} {n:4d} tok  {row['tok_s']:6.1f} tok/s   "
          f"accept {per_cycle:.2f}/5 = {100*per_cycle/5:4.1f}%   cycles {int(dd)}", flush=True)

if rows:
    best = max(rows, key=lambda r: r["tok_s"])
    print(f"\npeak {best['tok_s']} tok/s on {best['workload']}, "
          f"mean {round(sum(r['tok_s'] for r in rows)/len(rows), 1)} tok/s across {len(rows)} shapes")
if OUT:
    json.dump(rows, open(OUT, "w"), indent=1)
    print("wrote", OUT)
