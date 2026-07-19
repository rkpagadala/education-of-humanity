#!/bin/bash
# =============================================================================
# scripts/ml/cloud/startup.sh
#
# VM startup script. Runs as root on first boot of a preemptible T4 VM.
# Pulls the repo (from a GCS-staged tarball; no git auth needed inside VM),
# installs Python deps, launches the assigned job, uploads results to GCS,
# then shuts the VM down.
#
# Metadata keys the VM reads:
#   job-id          — unique job ID (used for output naming)
#   job-kind        — one of: smoke, transformer, spec_curve, placebos, ...
#   job-params      — JSON-encoded parameter dict
#   gcs-code-uri    — gs://bucket/path/to/repo_snapshot.tar.gz
#   gcs-output-uri  — gs://bucket/path/to/results/ (per-job subdir)
#   shutdown-on-completion — "true" to self-delete on success
#
# NOTE: We deliberately DO NOT set `set -e` — we want the cleanup/upload/
# self-delete steps to always run, even when the job code fails. Errors
# are captured and uploaded for diagnosis.
# =============================================================================
set -uo pipefail   # NB: no -e
exec > >(tee -a /var/log/job.log) 2>&1

trap 'on_exit' EXIT
on_exit() {
    local rc=$?
    echo "[$(date -Iseconds)] trap on_exit rc=$rc"
    # Best-effort upload of whatever we have
    if [ -d "/opt/job/output" ]; then
        cp /var/log/job.log /opt/job/output/job_full.log 2>/dev/null || true
        cp /opt/job/job_stdout.log /opt/job/output/job_stdout.log 2>/dev/null || true
        echo "$rc" > /opt/job/output/exit_code.txt
        if [ -n "${GCS_OUT:-}" ]; then
            gsutil -q -m cp -r /opt/job/output/* "$GCS_OUT" || true
        fi
    fi
    # Self-delete regardless of success — preempt budget protection
    if [ "${SHUTDOWN_DONE:-true}" = "true" ]; then
        local instance
        local zone
        instance=$(curl -s -H 'Metadata-Flavor: Google' \
            http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null)
        zone=$(curl -s -H 'Metadata-Flavor: Google' \
            http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null \
            | awk -F/ '{print $NF}')
        if [ -n "$instance" ] && [ -n "$zone" ]; then
            echo "[$(date -Iseconds)] self-deleting $instance in $zone"
            gcloud --quiet compute instances delete "$instance" \
                --zone="$zone" 2>&1 || true
        fi
    fi
}

echo "[$(date -Iseconds)] startup begin"

# Metadata
META="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
JOB_ID=$(curl -s -H 'Metadata-Flavor: Google' "$META/job-id")
JOB_KIND=$(curl -s -H 'Metadata-Flavor: Google' "$META/job-kind")
JOB_PARAMS=$(curl -s -H 'Metadata-Flavor: Google' "$META/job-params")
GCS_CODE=$(curl -s -H 'Metadata-Flavor: Google' "$META/gcs-code-uri")
GCS_OUT=$(curl -s -H 'Metadata-Flavor: Google' "$META/gcs-output-uri")
SHUTDOWN_DONE=$(curl -s -H 'Metadata-Flavor: Google' "$META/shutdown-on-completion" || echo "true")
# Generation-unification flag (set by launch.py from the launching shell's
# ML_UNIFY_GENERATION). The data_loader reads this to choose 28 vs the legacy
# 25-year grid. Default "0" preserves legacy behavior if the key is absent.
ML_UNIFY_GENERATION=$(curl -s -H 'Metadata-Flavor: Google' "$META/ml-unify-generation" || echo "0")

echo "[$(date -Iseconds)] job_id=$JOB_ID kind=$JOB_KIND"
echo "[$(date -Iseconds)] code=$GCS_CODE out=$GCS_OUT"
echo "[$(date -Iseconds)] ML_UNIFY_GENERATION=$ML_UNIFY_GENERATION"

# Make GCS_OUT and SHUTDOWN_DONE visible to the EXIT trap; ML_UNIFY_GENERATION
# visible to the job subprocess.
export GCS_OUT SHUTDOWN_DONE ML_UNIFY_GENERATION

# Working dir
WORK=/opt/job
mkdir -p "$WORK" && cd "$WORK"

# Pull code snapshot
gsutil -q cp "$GCS_CODE" repo.tar.gz
mkdir -p repo && tar -xzf repo.tar.gz -C repo --strip-components=1
cd repo

# Install Python deps (the deep-learning AMI has PyTorch+CUDA preinstalled).
# xlrd is needed for broader_features (Polity 5 .xls), openpyxl for Maddison.
pip install --quiet --upgrade pip
pip install --quiet pandas numpy scipy scikit-learn statsmodels matplotlib \
    seaborn xlrd openpyxl

# Run the job
echo "[$(date -Iseconds)] dispatching job kind=$JOB_KIND"
python3 scripts/ml/cloud/run_job.py \
    --job-id "$JOB_ID" \
    --kind "$JOB_KIND" \
    --params "$JOB_PARAMS" \
    --output-dir "$WORK/output" \
    2>&1 | tee "$WORK/job_stdout.log"
RC=$?
echo "[$(date -Iseconds)] job finished rc=$RC"

# The EXIT trap handles upload + self-delete. Nothing more to do here.
echo "[$(date -Iseconds)] startup script complete (rc=$RC)"
exit "$RC"
