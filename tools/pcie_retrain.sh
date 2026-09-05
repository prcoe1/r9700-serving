#!/usr/bin/env bash
# Force a PCIe link retrain on a root port (default: 0000:00:06.3, the
# card0 / GPU 08:00.0 chain on this box) to recover Gen5 after a silent
# Gen5->Gen4 downtrain (see SYSTEM.md §6).
#
# Two mechanisms, tried in order:
#   1. soft: setpci Link Control — set Target Link Speed + Change Link Speed
#      bit. Link retrains at the target speed; falls back if training fails.
#      Brief link interruption; devices behind the port see a hiccup.
#   2. hard (opt-in, --rescan): sysfs remove + rescan of the root port.
#      Full re-enumeration. Everything behind the port (the GPU!) is
#      torn down and re-added — vLLM will lose that GPU. Stop the server
#      (or at least accept a crash) before using this.
#
# Usage:
#   sudo pcie_retrain.sh                     # soft retrain on 0000:00:06.3, target Gen5
#   sudo pcie_retrain.sh 0000:00:06.0        # other root port
#   sudo pcie_retrain.sh --gen 4             # target Gen4 (verify stable fallback)
#   sudo pcie_retrain.sh --rescan            # fall back to remove/rescan if soft fails
#   pcie_retrain.sh --dry-run                # read-only state dump, no writes
#
# Root is required for any write (setpci / sysfs).
set -euo pipefail

BDF="0000:00:06.3"
GEN=5
RESCAN=0
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --rescan) RESCAN=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --gen) GEN="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    0*) BDF="$1"; shift ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

case "$GEN" in
  4) CODE=4; TARGET="16.0 GT/s PCIe" ;;
  5) CODE=5; TARGET="32.0 GT/s PCIe" ;;
  *) echo "--gen must be 4 or 5" >&2; exit 2 ;;
esac

[ -e "/sys/bus/pci/devices/$BDF" ] || { echo "no such PCI device: $BDF" >&2; exit 1; }
if [ "$DRY" = 0 ] && [ "$(id -u)" != 0 ]; then
  echo "root required (or use --dry-run)" >&2
  exit 1
fi

die() { echo "FAIL: $*" >&2; exit 1; }

lnkcap()  { # 4-byte reg at a 2-byte-aligned offset: read as two words
  local lo hi
  lo=$(setpci -s "$BDF" CAP_EXP+0x0e.w)
  hi=$(setpci -s "$BDF" CAP_EXP+0x10.w)
  printf '%04x%04x' "0x$hi" "0x$lo"
}
lnkctl()  { setpci -s "$BDF" CAP_EXP+0x12.w; }   # Link Control
lnksta()  { setpci -s "$BDF" CAP_EXP+0x14.l; }   # Link Status
lnkctl2() { setpci -s "$BDF" CAP_EXP+0x2e.w; }   # Link Control 2
cur_speed() { cat "/sys/bus/pci/devices/$BDF/current_link_speed"; }
cur_width() { cat "/sys/bus/pci/devices/$BDF/current_link_width"; }

aer_total() {
  local d="/sys/bus/pci/devices/$BDF" f v=0
  [ -d "$d" ] || { echo n/a; return; }
  for f in "$d"/aer_dev_correctable "$d"/aer_dev_nonfatal "$d"/aer_dev_fatal; do
    [ -r "$f" ] || continue
    local t; t=$(grep TOTAL_ERR "$f" | awk '{s+=$NF} END{print s+0}')
    v=$((v + t))
  done
  echo "$v"
}

print_state() {
  local tag="$1" lc lc2 st cap have_cap=0
  # config space past byte 64 (where PCIe caps live) is root-only
  command -v setpci >/dev/null 2>&1 && \
    setpci -s "$BDF" CAP_EXP+0x00.w >/dev/null 2>&1 && have_cap=1
  if [ "$have_cap" = 1 ]; then
    lc=$(lnkctl); lc2=$(lnkctl2); st=$(lnksta); cap=$(lnkcap)
    printf '%s: LnkSta=%s (speed=%s width=%s)  LnkCtl=%s  LnkCtl2=%s  LnkCap=%s  AER=%s\n' \
      "$tag" "$st" "$(cur_speed)" "$(cur_width)" "$lc" "$lc2" "$cap" "$(aer_total)"
  else
    printf '%s: speed=%s width=%s AER=%s (LnkSta/LnkCtl need root)\n' \
      "$tag" "$(cur_speed)" "$(cur_width)" "$(aer_total)"
  fi
}

wait_for_speed() {
  local target="$1" i
  for i in $(seq 1 20); do
    [ "$(cur_speed)" = "$target" ] && return 0
    sleep 0.5
  done
  return 1
}

echo "pcie retrain: $BDF -> target $GEN ($TARGET)"
print_state "before"
[ "$DRY" = 1 ] && exit 0

command -v setpci >/dev/null 2>&1 || die "setpci not found (apt: pciutils)"

# --- step 1: soft retrain via Link Control ---------------------------------
cur=$(lnkctl)
cap=$(lnkcap)
cap_dec=$((16#$cap))
sup=$(( (cap_dec & 0x1f0) >> 4 ))        # Supported Link Speeds vector
if [ "$sup" -lt "$CODE" ]; then
  die "port does not advertise Gen$GEN (Supported Link Speeds = 0x$(printf %x "$sup"))"
fi
# Target Link Speed = bits [2:0], Change Link Speed = bit 3
new=$(( (0x$cur & ~0x0F) | CODE | 8 ))
echo "LnkCtl: 0x$cur -> 0x$(printf %04x "$new") (target=Gen$GEN, change-bit set)"
setpci -s "$BDF" CAP_EXP+0x12.w=$(printf %04x "$new")
# retrain-link bit in LnkCtl2 (belt and braces; spec says Change Link Speed
# alone suffices for a speed-change retrain)
c2=$(lnkctl2)
setpci -s "$BDF" CAP_EXP+0x2e.w=$(printf %04x $(( (0x$c2 | 0x800) )))

if wait_for_speed "$TARGET"; then
  echo "OK: link at $TARGET"
  print_state "after"
  exit 0
fi

echo "soft retrain did not reach $TARGET (still $(cur_speed))"
print_state "after-soft"

# --- step 2: hard remove/rescan (opt-in) ------------------------------------
if [ "$RESCAN" = 1 ]; then
  echo
  echo "WARNING: removing $BDF tears down everything behind it"
  echo "(GPU $BDF-chain). Any workload using that GPU will crash."
  read -r -p "Continue? [y/N] " ans
  [ "$ans" = "y" ] || { echo "aborted"; exit 1; }
  echo 1 > "/sys/bus/pci/devices/$BDF/remove"
  sleep 1
  echo 1 > /sys/bus/pci/rescan
  if wait_for_speed "$TARGET"; then
    echo "OK: link at $TARGET after rescan"
  else
    echo "still $(cur_speed) after rescan"
  fi
  print_state "after-rescan"
else
  echo "use --rescan to attempt a full remove/rescan (drops the GPU behind this port)"
fi
exit 0
