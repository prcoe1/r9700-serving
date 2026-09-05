#!/usr/bin/env bash
# Check PCIe link state for the R9700 GPUs and their upstream chain.
# No root needed (reads /sys only). Speed is the health signal; sysfs width
# reads 16 even on the x8 root ports, so width is reference-only.
#
# Usage:
#   pcie_link_check.sh            # one-shot table
#   pcie_link_check.sh --aer      # one-shot table + AER counters
#   pcie_link_check.sh --watch [s]# log state changes every s seconds
#                                  # (default 5s) to WATCH_LOG
#   WATCH_LOG=/path/to.log ...     # override log location
set -euo pipefail

WATCH=0
AER=0
INTERVAL=5
WATCH_LOG="${WATCH_LOG:-${HOME}/.cache/pcie-link-watch.log}"

while [ $# -gt 0 ]; do
  case "$1" in
    --watch) WATCH=1; shift ;;
    --aer) AER=1; shift ;;
    --help|-h)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    ''|*)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        INTERVAL="$1"; WATCH=1
      else
        echo "unknown argument: $1 (see --help)" >&2
        exit 2
      fi
      shift ;;
  esac
done

GEN5="32.0 GT/s PCIe"

dev_speed() { cat "/sys/bus/pci/devices/$1/current_link_speed" 2>/dev/null || echo n/a; }
dev_width() { cat "/sys/bus/pci/devices/$1/current_link_width" 2>/dev/null || echo n/a; }
dev_up() {
  local bdf="$1" up
  up=$(basename "$(dirname "/sys/bus/pci/devices/$bdf")")
  [ "$up" = "0000" ] && echo "root" || echo "$up"
}

# rows: "label|pci-bdf"
# Topology (verified 2026-09-04):
#   card0 / 08:00.0 : rp 00:06.3 -> sw 06:00.0 -> 07:00.0 -> gpu
#   card1 / 04:00.0 : rp 00:06.0 -> sw 02:00.0 -> 03:00.0 -> gpu
ROWS=(
  "card0 (GPU 08:00.0)|0000:08:00.0"
  "  sw 06:00.0|0000:06:00.0"
  "  rp 00:06.3|0000:00:06.3"
  "card1 (GPU 04:00.0)|0000:04:00.0"
  "  sw 02:00.0|0000:02:00.0"
  "  rp 00:06.0|0000:00:06.0"
)

fmt_row() {
  local label="$1" bdf="$2"
  local s w
  s=$(dev_speed "$bdf"); w=$(dev_width "$bdf")
  if [ "$s" = "$GEN5" ]; then
    printf '%-20s %-18s x%s\n' "$label" "$s" "$w"
  else
    printf '%-20s \033[31m%-18s x%s\033[0m\n' "$label" "$s" "$w"
  fi
}

aer_total() {
  local d="/sys/bus/pci/devices/$1" f v=0
  [ -d "$d" ] || { echo n/a; return; }
  for f in "$d"/aer_dev_correctable "$d"/aer_dev_nonfatal "$d"/aer_dev_fatal; do
    [ -r "$f" ] || continue
    local t; t=$(grep TOTAL_ERR "$f" | awk '{s+=$NF} END{print s+0}')
    v=$((v + t))
  done
  echo "$v"
}

print_table() {
  echo "PCIe link state  ($(date '+%Y-%m-%d %H:%M:%S'))  — health target: $GEN5"
  local r label bdf
  for r in "${ROWS[@]}"; do
    IFS='|' read -r label bdf <<<"$r"
    fmt_row "$label" "$bdf"
  done
  if [ "$AER" = 1 ]; then
    echo
    echo "AER counter totals (correctable+nonfatal+fatal):"
    local bdfs="0000:08:00.0 0000:02:00.0 0000:00:06.0 0000:04:00.0 0000:06:00.0 0000:00:06.3"
    for bdf in $bdfs; do
      printf '  %-14s %s\n' "$bdf" "$(aer_total "$bdf")"
    done
  fi
}

if [ "$WATCH" = 0 ]; then
  print_table
  exit 0
fi

state_sig() {
  local r bdf
  for r in "${ROWS[@]}"; do
    IFS='|' read -r _ bdf <<<"$r"
    printf '%s:' "$bdf"
    dev_speed "$bdf" | tr -d ' \n'
  done
}

last_sig=$(state_sig)
echo "watching every ${INTERVAL}s; log: $WATCH_LOG"
echo "$(date '+%F %T') watch start (interval=${INTERVAL}s); $(printf '%s' "$last_sig" | tr ':' '\n' | paste -sd' ')" >>"$WATCH_LOG"

while :; do
  sleep "$INTERVAL"
  sig=$(state_sig)
  if [ "$sig" != "$last_sig" ]; then
    ts=$(date '+%F %T')
    old=$(printf '%s' "$last_sig" | tr ':' '\n' | paste -sd' ')
    new=$(printf '%s' "$sig" | tr ':' '\n' | paste -sd' ')
    echo "$ts CHANGED: $old -> $new" >>"$WATCH_LOG"
    echo "$ts CHANGED:"
    printf '  old: %s\n  new: %s\n' "$old" "$new"
    print_table
    last_sig="$sig"
  fi
done
