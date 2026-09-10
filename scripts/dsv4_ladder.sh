#!/usr/bin/env bash
# dsv4_ladder.sh <recipe.yaml>...
# Fresh boot per recipe, then two llama-benchy grids against it: text (default book) and code
# (a 130 KB Python source file), each at depths 0..32k and concurrency 1/2/5, pp 2048, tg 128,
# prefix caching on so depth contexts are served the way a real agent session serves them.
# Speculative counters are read around each grid so acceptance is attributed per lane.
# A boot that dies in the worker rendezvous (the intermittent race) is retried once.
export PATH="$HOME/.local/bin:$PATH"
R="$HOME/GEN-AI/DeepSeek-v4-Flash-DSpark-DGX-Spark-TP2"; cd "$R" || exit 1
OUT="${LADDER_OUT:-$R/results/ladder}"; mkdir -p "$OUT"; LOG="$OUT/ladder.log"
BASE=http://192.168.100.62:8010; WORKER=192.168.100.53
TOK=$(ls -d ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/*/ | head -1)
DEPTHS=${DEPTHS:-"0 4096 8192 16384 32768"}; CONC=${CONC:-"1 2 5"}; RUNS=${RUNS:-2}
CODE_URL=${CODE_URL:-https://raw.githubusercontent.com/python/cpython/3.13/Lib/typing.py}
say() { echo "=== $(date +%T) $*" | tee -a "$LOG"; }
spec() { curl -s -m 6 "$BASE/metrics" | grep -E '^vllm:spec_decode_num_(accepted_tokens|drafts)_total' | awk '{s[$1]+=$2} END{for(k in s) print k, s[k]}'; }
accept() { paste <(sort "$1") <(sort "$2") | awk '{split($1,a,"{"); d[a[1]]=$4-$2} END{if(d["vllm:spec_decode_num_drafts_total"]>0) printf "accept/cycle %.2f over %d cycles", d["vllm:spec_decode_num_accepted_tokens_total"]/d["vllm:spec_decode_num_drafts_total"], d["vllm:spec_decode_num_drafts_total"]}'; }

boot() {  # boot <recipe> <name> -> 0 healthy, 1 failed, 2 rendezvous race (retry-worthy)
  sparkrun stop --all >/dev/null 2>&1; sleep 5
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  ssh "$WORKER" 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null'
  sparkrun run "$1" --cluster dgx-cluster-cx7 --tp 2 --trust --no-follow > "$OUT/$2-sparkrun.log" 2>&1 || { say "$2 sparkrun exit"; grep -aE "failed|Error" "$OUT/$2-sparkrun.log" | tail -2 | tee -a "$LOG"; return 1; }
  for _ in $(seq 1 90); do
    [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' $BASE/health)" = 200 ] && return 0
    C=$(docker ps -q | head -1); [ -z "$C" ] && break
    docker exec "$C" grep -aqE "Engine core initialization failed|WorkerProc failed to start" /tmp/sparkrun_serve.log 2>/dev/null && break
    sleep 20
  done
  C=$(docker ps -q | head -1); [ -n "$C" ] && docker exec "$C" cat /tmp/sparkrun_serve.log > "$OUT/$2-serve.log" 2>/dev/null
  grep -aqE "Timed out after [0-9]+ seconds waiting for clients|FileNotFoundError.*No such file" "$OUT/$2-serve.log" 2>/dev/null && return 2
  return 1
}

i=0
for rec in "$@"; do
  i=$((i+1)); name=$(printf "%02d-%s" "$i" "$(basename "$rec" .yaml)"); say "$name start"
  boot "$rec" "$name"; rc=$?
  if [ $rc = 2 ]; then say "$name rendezvous race, retrying once"; boot "$rec" "$name"; rc=$?; fi
  if [ $rc != 0 ]; then say "$name never healthy"; grep -aE "Error|error:" "$OUT/$name-serve.log" 2>/dev/null | grep -viE "use_fast|deprecated|core.py:1231" | tail -2 | cut -c1-200 | tee -a "$LOG"; sparkrun stop --all >/dev/null 2>&1; continue; fi
  C=$(docker ps -q | head -1); docker exec "$C" cat /tmp/sparkrun_serve.log > "$OUT/$name-serve.log" 2>/dev/null
  kv=$(grep -aoE "GPU KV cache size: [0-9,]+ tokens" "$OUT/$name-serve.log" | head -1)
  mem="avail head=$(free -g | awk '/^Mem:/{print $7}')G worker=$(ssh "$WORKER" "free -g | awk '/^Mem:/{print \$7}'")G"
  M=$(curl -s -m 6 $BASE/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
  say "$name healthy | $kv | $mem"
  for lane in text code; do
    book=(); [ $lane = code ] && book=(--book-url "$CODE_URL")
    spec > "$OUT/$name-$lane-spec0.txt"
    uvx llama-benchy@0.4.0 --base-url "$BASE/v1" --model "$M" --tokenizer "$TOK" \
      --depth $DEPTHS --concurrency $CONC --pp 2048 --tg 128 --runs "$RUNS" --enable-prefix-caching \
      --extra-body '{"chat_template_kwargs":{"thinking":false}}' "${book[@]}" \
      --save-result "$OUT/$name-$lane.csv" > "$OUT/$name-$lane-benchy.log" 2>&1
    spec > "$OUT/$name-$lane-spec1.txt"
    say "$name $lane done | $(accept "$OUT/$name-$lane-spec0.txt" "$OUT/$name-$lane-spec1.txt")"
    awk -F'|' '$3 ~ /tg128/ {gsub(/ /,"",$3); gsub(/ /,"",$4); gsub(/ /,"",$5); print "   " $3, "total=" $4, "req=" $5}' "$OUT/$name-$lane.csv" | tee -a "$LOG"
  done
  [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' $BASE/health)" = 200 ] || say "!!! $name unhealthy after grids"
  say "$name done"; sparkrun stop --all >/dev/null 2>&1
done
say LADDER_DONE
