#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash with DSpark at TP=2 across two DGX Sparks with sparkrun.
# Run on the head node. Every recipe in this repository is one option below.
#
#   scripts/run.sh <option> [--check] [--bench] [--skip-download]
#
#   option              recipe                                                   engine
#   vision     recipes/dsv4-vision-dspark-tp2.yaml          images + text, 400k, fp8 KV
#   vision-1m  recipes/dsv4-vision-dspark-tp2-nvfp4kv.yaml  images + text, 1M, NVFP4 sparse-MLA KV
#   text       recipes/dsv4-text-dspark-tp2.yaml            text only, 1M, largest KV pool
#
#   --check          validate cluster and nodes, launch nothing
#   --bench          after the server is up, run the llama-benchy grid (DEPTHS, CONCURRENCY env)
#   --skip-download  checkpoint already on both nodes
#
# The recipes carry no bind mounts; the image applies its own hotfixes at startup.
# Adjust the fabric names first with scripts/detect-fabric.sh, and disable earlyoom on both nodes.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPTION="${1:-}"; shift || true
CLUSTER="${CLUSTER:-dgx-cluster-cx7}"
HEAD_IP="${HEAD_IP:-192.168.100.62}"
WORKER_IP="${WORKER_IP:-192.168.100.53}"
BENCH=0; SKIP_DOWNLOAD=0; CHECK_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --bench) BENCH=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --check) CHECK_ONLY=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

case "$OPTION" in
  vision)     ENGINE=vllm; RECIPE="$REPO/recipes/dsv4-vision-dspark-tp2.yaml" ;;
  vision-1m)  ENGINE=vllm; RECIPE="$REPO/recipes/dsv4-vision-dspark-tp2-nvfp4kv.yaml" ;;
  text)       ENGINE=vllm; RECIPE="$REPO/recipes/dsv4-text-dspark-tp2.yaml" ;;
  *) sed -n '2,16p' "$0"; exit 2 ;;
esac

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

META=$(python3 "$REPO/scripts/recipe_metadata.py" "$RECIPE" "$ENGINE") || fail "invalid recipe metadata"
IFS='|' read -r MODEL REVISION SERVED PORT _ _ <<< "$META"
read -r -a DEPTH_GRID <<< "${DEPTHS:-0 16384}"
read -r -a CONC_GRID <<< "${CONCURRENCY:-1 2}"
EXTRA=()

say "Checking prerequisites"
command -v docker >/dev/null || fail "docker not found"
SPARKRUN="${SPARKRUN:-$HOME/.local/bin/sparkrun}"
[ -x "$SPARKRUN" ] || fail "sparkrun not found at $SPARKRUN"
command -v uvx >/dev/null || echo "  warning: uvx missing, --bench will not work"

say "Checking cluster '$CLUSTER'"
"$SPARKRUN" cluster list 2>/dev/null | grep -q "$CLUSTER" \
  || fail "cluster '$CLUSTER' not defined. Create it with: sparkrun cluster create $CLUSTER --hosts $HEAD_IP,$WORKER_IP"
HOSTS=$("$SPARKRUN" cluster show "$CLUSTER" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | sort -u)
[ "$(echo "$HOSTS" | wc -l | tr -d ' ')" -eq 2 ] || fail "TP=2 needs exactly 2 hosts, cluster has: $(echo $HOSTS)"
for ip in $HOSTS; do
  case "$ip" in 192.168.100.*) ;; *) fail "host $ip is not on the CX-7 subnet; collectives would run over WiFi" ;; esac
done
echo "  ok: 2 hosts on CX-7 ($(echo $HOSTS | tr '\n' ' '))"

say "Checking both nodes"
for ip in $HOSTS; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$ip" true 2>/dev/null || fail "no passwordless SSH to $ip"
  ssh "$ip" 'command -v docker >/dev/null' || fail "$ip: docker not installed"
  ssh "$ip" "[ -x \"\$HOME/.local/bin/sparkrun\" ]" || fail "$ip: sparkrun not installed"
  MEM=$(ssh "$ip" "free -g | awk '/^Mem:/{print \$2}'"); [ "$MEM" -ge 100 ] || fail "$ip: ${MEM}G total memory, expected ~121G"
  ssh "$ip" 'systemctl is-active --quiet earlyoom' && echo "  WARNING: $ip runs earlyoom; it can kill the engine under deep-context load"
  BUSY=$(ssh "$ip" 'docker ps -q --filter name=sparkrun | wc -l'); [ "$BUSY" -eq 0 ] || fail "$ip: a sparkrun container is running. Stop it first: sparkrun stop --all"
  echo "  ok: $ip  mem=${MEM}G"
done
ssh "$HEAD_IP" "ping -c 2 -W 3 $WORKER_IP" >/dev/null 2>&1 || fail "$HEAD_IP cannot reach $WORKER_IP over CX-7"
echo "  ok: CX-7 path $HEAD_IP -> $WORKER_IP"

if [ "$CHECK_ONLY" = 1 ]; then say "Checks passed, not launching"; exit 0; fi

say "Materialising recipe for $REPO"
GEN="$REPO/.run"; mkdir -p "$GEN"; RECIPE_OUT="$GEN/$(basename "$RECIPE")"
python3 "$REPO/scripts/recipe_metadata.py" "$RECIPE" "$ENGINE" --render > "$RECIPE_OUT"
echo "  ok: $RECIPE_OUT"

say "Syncing repo to worker at the same path"
ssh "$WORKER_IP" "mkdir -p '$REPO'"
rsync -a --delete --exclude '.git' "$REPO/" "$WORKER_IP:$REPO/"

if [ "$SKIP_DOWNLOAD" = 0 ]; then
  say "Fetching checkpoint on head, mirroring to worker over CX-7"
  d="$HOME/.cache/huggingface/hub/models--${MODEL//\//--}"
  hf download "$MODEL" --revision "$REVISION"
  mkdir -p "$d/refs" && printf '%s' "$REVISION" > "$d/refs/main"
  rsync -a -e 'ssh -o Compression=no' "$d/" "$WORKER_IP:$d/"
fi

say "Dropping page cache on both nodes"
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
ssh "$WORKER_IP" 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null'

say "Launching $OPTION"
mkdir -p "$REPO/results"; LOG="$REPO/results/${OPTION}-serve.log"
export PATH="$HOME/.local/bin:$PATH"
nohup setsid "$SPARKRUN" run "$RECIPE_OUT" --cluster "$CLUSTER" --tp 2 > "$LOG" 2>&1 < /dev/null &
echo "  log: $LOG"

say "Waiting for the server (loads take 10-15 minutes)"
SEEN=0
for _ in $(seq 1 120); do
  [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://$HEAD_IP:$PORT/health")" = 200 ] && { READY=1; break; }
  if docker ps -q | grep -q .; then SEEN=1; elif [ "$SEEN" = 1 ]; then fail "container exited during load, see $LOG"; fi
  grep -aqE "launch_inference failed|Engine core initialization failed" "$LOG" 2>/dev/null && ! docker ps -q | grep -q . && fail "launch failed, see $LOG"
  printf '  %s  %sG available\n' "$(date +%H:%M:%S)" "$(free -g | awk '/^Mem:/{print $7}')"; sleep 30
done
[ "${READY:-0}" = 1 ] || fail "server never became ready, see $LOG"
say "Server up at http://$HEAD_IP:$PORT"

if [ "$BENCH" = 1 ]; then
  say "Benchmarking"
  uvx llama-benchy@0.4.0 --base-url "http://$HEAD_IP:$PORT/v1" --model "$SERVED" --tokenizer "$MODEL" \
    "${EXTRA[@]}" --depth "${DEPTH_GRID[@]}" --pp 2048 --tg 128 --enable-prefix-caching \
    --concurrency "${CONC_GRID[@]}" --save-result "$REPO/results/${OPTION}-run.csv"
fi
say "Done. Stop with: sparkrun stop --all"
