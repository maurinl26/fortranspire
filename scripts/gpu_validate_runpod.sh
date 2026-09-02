#!/usr/bin/env bash
# gpu_validate_runpod.sh — validate a Phase-1 GPU port on an on-demand RunPod GPU.
#
# The dev machine has no NVIDIA GPU, so the port is only generated. This spins up
# a RunPod GPU pod running the fortranspire-nvhpc image (issue #121), sends the
# generated output/, runs `gpu_validate.sh` there (nvfortran compile + the
# equivalence harness), reports the numerical verdict, then DESTROYS the pod.
#
#   scripts/gpu_validate_runpod.sh output/            # real run (needs a pod)
#   scripts/gpu_validate_runpod.sh --dry-run output/  # print the plan, no pod
#
# ⚠️ SOVEREIGNTY: RunPod is a US cloud. Fine for test/CI; NOT for porting client
#    code under a sovereignty constraint — use a sovereign GPU node instead
#    (FORTRANSPIRE_GPU_HOST / EWC #43) with the same nvfortran image.
#
# Requirements: runpodctl (+ RUNPOD_API_KEY), ssh, rsync. The exact runpodctl
# flags are isolated below and configurable — bump them per your runpodctl
# version if a step fails; the teardown is guaranteed regardless.
set -euo pipefail

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && { DRY_RUN=1; shift; }
OUT="${1:-output}"

# ── Config (override via env) ────────────────────────────────────────────────
IMAGE="${FORTRANSPIRE_NVHPC_IMAGE:-ghcr.io/maurinl26/fortranspire-nvhpc:latest}"
GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA A40}"
DISK_GB="${RUNPOD_DISK_GB:-20}"
POD_NAME="fortranspire-gpu-$$"
MAX_WAIT="${RUNPOD_MAX_WAIT:-600}"        # seconds to wait for the pod to be ready

echo "=== fortranspire GPU validation on RunPod ==="
echo "  image    : ${IMAGE}"
echo "  gpu      : ${GPU_TYPE}   disk ${DISK_GB}G"
echo "  output   : ${OUT}"
echo "  ⚠️  RunPod is a US cloud — validation/CI only, not sovereign porting."

# ── Preconditions ────────────────────────────────────────────────────────────
_need() { command -v "$1" >/dev/null 2>&1 || { echo "❌ missing: $1"; MISSING=1; }; }
MISSING=0; _need runpodctl; _need ssh; _need rsync
[ -d "${OUT}" ] || { echo "❌ no ${OUT}/ — run 'fortranspire gpu <kernel>' first"; MISSING=1; }
[ -n "${RUNPOD_API_KEY:-}" ] || { echo "❌ RUNPOD_API_KEY not set (runpod.io → settings → API keys)"; MISSING=1; }

if [ "${DRY_RUN}" = "1" ]; then
  echo ""
  echo "--- DRY RUN plan ---"
  echo "  1. runpodctl create pod --name ${POD_NAME} --imageName ${IMAGE} \\"
  echo "        --gpuType '${GPU_TYPE}' --gpuCount 1 --containerDiskSize ${DISK_GB} --ports '22/tcp'"
  echo "  2. poll 'runpodctl get pod <id>' until RUNNING + SSH (≤ ${MAX_WAIT}s)"
  echo "  3. rsync ${OUT}/ scripts/ → pod:/work/"
  echo "  4. ssh pod: 'cd /work && bash scripts/gpu_validate.sh ${OUT}'"
  echo "  5. runpodctl remove pod <id>   (ALSO on any failure — trap)"
  echo "  preconditions ok: $([ "${MISSING}" = "0" ] && echo yes || echo 'NO — see above')"
  exit $([ "${MISSING}" = "0" ] && echo 0 || echo 2)
fi
[ "${MISSING}" = "0" ] || exit 2

# ── 1. Create the pod ────────────────────────────────────────────────────────
echo "  [1/5] Creating pod …"
CREATE_OUT="$(runpodctl create pod --name "${POD_NAME}" --imageName "${IMAGE}" \
    --gpuType "${GPU_TYPE}" --gpuCount 1 --containerDiskSize "${DISK_GB}" \
    --ports '22/tcp')"
echo "${CREATE_OUT}"
POD_ID="$(echo "${CREATE_OUT}" | grep -oE 'pod "[^"]+"' | head -1 | sed 's/pod //; s/"//g')"
[ -n "${POD_ID}" ] || { echo "❌ could not parse pod id from runpodctl output"; exit 3; }

# GUARANTEED teardown — the one thing that must never be skipped (cost).
trap 'echo "  [5/5] Destroying pod ${POD_ID} …"; runpodctl remove pod "${POD_ID}" || true' EXIT

# ── 2. Wait until RUNNING with SSH exposed ───────────────────────────────────
echo "  [2/5] Waiting for pod ${POD_ID} to be ready (≤ ${MAX_WAIT}s) …"
deadline=$(( $(date +%s) + MAX_WAIT ))
SSH_HOST=""; SSH_PORT=""
while [ "$(date +%s)" -lt "${deadline}" ]; do
  INFO="$(runpodctl get pod "${POD_ID}" --allfields 2>/dev/null || true)"
  # Public SSH endpoint: "<ip>:<port>->22". Parse defensively.
  MAP="$(echo "${INFO}" | grep -oE '[0-9.]+:[0-9]+->22' | head -1 || true)"
  if echo "${INFO}" | grep -qi 'RUNNING' && [ -n "${MAP}" ]; then
    SSH_HOST="${MAP%%:*}"; SSH_PORT="$(echo "${MAP}" | sed 's/.*://; s/->22//')"
    break
  fi
  sleep 10
done
[ -n "${SSH_HOST}" ] && [ -n "${SSH_PORT}" ] || { echo "❌ pod not ready / no SSH endpoint"; exit 4; }
echo "        ssh root@${SSH_HOST} -p ${SSH_PORT}"

SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "${SSH_PORT}" "root@${SSH_HOST}")
for _ in $(seq 1 30); do "${SSH[@]}" true 2>/dev/null && break; sleep 5; done

# ── 3. Send the port + scripts ───────────────────────────────────────────────
echo "  [3/5] Sending ${OUT}/ and scripts/ …"
rsync -az -e "ssh -o StrictHostKeyChecking=no -p ${SSH_PORT}" \
    "${OUT}/" "scripts/" "root@${SSH_HOST}:/work/"

# ── 4. Compile with nvfortran AND run the equivalence harness ────────────────
echo "  [4/5] Validating on the GPU (nvfortran compile + equivalence harness) …"
"${SSH[@]}" "cd /work && bash scripts/gpu_validate.sh ${OUT}"
RC=$?

echo ""
[ "${RC}" = "0" ] && echo "✅ GPU-VALIDATED on RunPod (output matches the original Fortran)." \
                  || echo "❌ GPU validation FAILED (rc=${RC}) — see the harness output above."
exit "${RC}"
