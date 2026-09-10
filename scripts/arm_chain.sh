#!/usr/bin/env bash
# arm_chain.sh <recipe.yaml>...   Fresh boot per recipe, 3-trial measurement, stop. Results in
# results/arms/<name>.json plus a one-line summary per arm in results/arms/chain.log. A control
# (the baseline recipe) belongs in the list at least twice so drift brackets the arms.
export PATH="$HOME/.local/bin:$PATH"
R="$HOME/GEN-AI/DeepSeek-v4-Flash-DSpark-DGX-Spark-TP2"; cd "$R" || exit 1
OUT="$R/results/arms"; mkdir -p "$OUT"; LOG="$OUT/chain.log"
BASE=http://192.168.100.62:8010; WORKER=192.168.100.53
say() { echo "=== $(date +%T) $*" | tee -a "$LOG"; }

for rec in "$@"; do
  name=$(basename "$rec" .yaml)
  say "$name start"
  sparkrun stop --all >/dev/null 2>&1; sleep 5
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  ssh "$WORKER" 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null'
  sparkrun run "$rec" --cluster dgx-cluster-cx7 --tp 2 --trust > "$OUT/$name-sparkrun.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then say "$name sparkrun exit $rc"; grep -aE "failed|Error" "$OUT/$name-sparkrun.log" | tail -3 | tee -a "$LOG"; continue; fi
  up=0
  for _ in $(seq 1 90); do
    [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' $BASE/health)" = 200 ] && { up=1; break; }
    C=$(docker ps -q | head -1)
    [ -z "$C" ] && break
    docker exec "$C" grep -aqE "Engine core initialization failed|WorkerProc failed to start" /tmp/sparkrun_serve.log 2>/dev/null && break
    sleep 20
  done
  C=$(docker ps -q | head -1)
  [ -n "$C" ] && docker exec "$C" cat /tmp/sparkrun_serve.log > "$OUT/$name-serve.log" 2>/dev/null
  if [ $up != 1 ]; then
    say "$name never healthy"; grep -aE "Error|error:" "$OUT/$name-serve.log" | grep -viE "use_fast|deprecated|core.py:1231" | tail -3 | cut -c1-200 | tee -a "$LOG"
    sparkrun stop --all >/dev/null 2>&1; continue
  fi
  kv=$(grep -aoE "GPU KV cache size: [0-9,]+ tokens" "$OUT/$name-serve.log" | head -1)
  M=$(curl -s -m 6 $BASE/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
  python3 scripts/measure.py $BASE "$M" "$OUT/$name.json" --trials 3 > "$OUT/$name-measure.log" 2>&1
  summary=$(python3 - "$OUT/$name.json" <<'PY'
import json,sys,statistics
rows=json.load(open(sys.argv[1])); d={r["workload"]:r for r in rows}
real=[r["decode_tok_s"] for r in rows if r["workload"]!="counting"]
print(" ".join(f"{k}={d[k]['decode_tok_s']}" for k in ("prose","code","json","table","counting") if k in d),
      f"| median_real={statistics.median(real):.1f} | prose_accept={d['prose']['accepted_per_cycle']}" if "prose" in d else "")
PY
)
  say "$name done | $kv | $summary"
  sparkrun stop --all >/dev/null 2>&1
done
say "CHAIN_DONE"
