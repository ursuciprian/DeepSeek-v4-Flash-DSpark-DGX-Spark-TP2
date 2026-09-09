#!/usr/bin/env bash
# Print the NCCL fabric values for THIS pair of nodes and, with --write, patch them into the recipes.
# The published recipes carry this cluster's names (enp1s0f1np1 / rocep1s0f1,roceP2p1s0f1); many DGX
# Spark pairs are wired on f0 instead. sparkrun's -o overrides do not reach `env:`, so the YAML must
# be edited. Run on the head node with the worker reachable.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER=${WORKER_IP:-}
[ -n "$WORKER" ] || { echo "set WORKER_IP=<worker ip on the fast fabric>"; exit 2; }
IFACE=$(ip -o -4 route get "$WORKER" 2>/dev/null | grep -oP 'dev \K\S+')
[ -n "$IFACE" ] || { echo "no route to $WORKER"; exit 1; }
HCAS=$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ',' | sed 's/,$//')
GID=3
echo "interface : $IFACE"
echo "IB HCAs   : ${HCAS:-none found}"
echo "GID index : $GID  (try 5 if RoCE fails to connect)"
[ -n "$HCAS" ] || echo "warning: no RDMA devices; NCCL will fall back to TCP and decode jitter rises"
if [ "${1:-}" = --write ]; then
  for f in "$REPO"/recipes/latest/*.yaml "$REPO"/recipes/sparkarena/*.yaml; do
    sed -i.bak -E "s/(NCCL_SOCKET_IFNAME|GLOO_SOCKET_IFNAME|TP_SOCKET_IFNAME): .*/\1: $IFACE/; s/NCCL_IB_HCA: .*/NCCL_IB_HCA: $HCAS/" "$f"
    echo "patched $(basename "$f")"
  done
  echo "originals kept as *.yaml.bak; also check the taskset CPU list in the SGLang command"
fi
