#!/bin/bash
# =============================================================================
# scripts/ml/cloud/startup_worker.sh
#
# Worker-pool VM startup. Unlike startup.sh (one job per VM, then delete),
# this boots ONCE, installs deps ONCE, then runs worker_loop.py which claims
# and runs many jobs from the shared GCS queue until it is drained. The VM
# self-deletes only when the loop returns (queue empty) or on preemption.
#
# Amortizing boot + driver install + pip across many jobs is the whole point:
# the ~5-6 min fixed startup cost is paid once per worker, not once per job.
#
# Metadata keys:
#   gcs-code-uri  — gs://bucket/path/to/repo_snapshot.tar.gz
#   bucket        — results/queue bucket name (no gs:// prefix)
#   worker-id     — unique worker label (for claim attribution)
# =============================================================================
set -uo pipefail   # NB: no -e — the EXIT trap must always run
exec > >(tee -a /var/log/job.log) 2>&1

trap 'on_exit' EXIT
on_exit() {
    local rc=$?
    echo "[$(date -Iseconds)] worker on_exit rc=$rc"
    # Upload the worker's own console log for diagnosis.
    if [ -n "${BUCKET:-}" ] && [ -n "${WORKER_ID:-}" ]; then
        gsutil -q cp /var/log/job.log \
            "gs://${BUCKET}/worker_logs/${WORKER_ID}.log" 2>/dev/null || true
    fi
    # Self-delete (queue drained or preemption shutdown).
    local instance zone
    instance=$(curl -s -H 'Metadata-Flavor: Google' \
        http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null)
    zone=$(curl -s -H 'Metadata-Flavor: Google' \
        http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null \
        | awk -F/ '{print $NF}')
    if [ -n "$instance" ] && [ -n "$zone" ]; then
        echo "[$(date -Iseconds)] self-deleting $instance in $zone"
        gcloud --quiet compute instances delete "$instance" --zone="$zone" 2>&1 || true
    fi
}

echo "[$(date -Iseconds)] worker startup begin"

META="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
GCS_CODE=$(curl -s -H 'Metadata-Flavor: Google' "$META/gcs-code-uri")
BUCKET=$(curl -s -H 'Metadata-Flavor: Google' "$META/bucket")
WORKER_ID=$(curl -s -H 'Metadata-Flavor: Google' "$META/worker-id")
# Generation-unification flag (set by launch.py). worker_loop.py runs each job
# as a subprocess that inherits this env, so exporting it here is sufficient.
# Default "0" keeps legacy 25-year-grid behavior if the key is absent.
ML_UNIFY_GENERATION=$(curl -s -H 'Metadata-Flavor: Google' "$META/ml-unify-generation" || echo "0")
VM_NAME=$(curl -s -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/name)
export BUCKET WORKER_ID VM_NAME ML_UNIFY_GENERATION

echo "[$(date -Iseconds)] worker=$WORKER_ID vm=$VM_NAME bucket=$BUCKET ML_UNIFY_GENERATION=$ML_UNIFY_GENERATION"

WORK=/opt/job
mkdir -p "$WORK" && cd "$WORK"

# Pull code snapshot once.
gsutil -q cp "$GCS_CODE" repo.tar.gz
mkdir -p repo && tar -xzf repo.tar.gz -C repo --strip-components=1
cd repo

# Install deps once (DLVM image already has PyTorch + CUDA).
pip install --quiet --upgrade pip
pip install --quiet pandas numpy scipy scikit-learn statsmodels matplotlib \
    seaborn xlrd openpyxl optuna

# Run the claim/run/upload loop until the queue is drained.
echo "[$(date -Iseconds)] entering worker loop"
python3 scripts/ml/cloud/worker_loop.py
RC=$?
echo "[$(date -Iseconds)] worker loop returned rc=$RC"
exit "$RC"
