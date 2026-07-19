# =============================================================================
# scripts/ml/cloud/queue_lib.py
#
# GCS-backed claim queue for the Chapter 9 worker pool.
#
# Layout under gs://<bucket>/:
#   queue/manifest.json        — full job list [{id, kind, params}, ...]
#   queue/claims/<id>.json     — atomic claim marker {worker, vm, job}
#   results/<id>/result.json   — done marker (existing convention)
#
# A job is AVAILABLE when it has no result.json (not done) and no claim file
# (not in progress). Claiming is race-free: we create the claim object with
# the precondition `x-goog-if-generation-match:0`, which the GCS API honors
# atomically — exactly one concurrent worker succeeds, the rest get HTTP 412.
#
# Preemption recovery: a worker that dies mid-job leaves a claim but no
# result.json. The orchestrator (launch.py worker-pool monitor) reaps such
# claims once the owning VM is gone, returning the job to the available pool.
# =============================================================================
"""queue_lib.py — race-free GCS job queue primitives (stdlib + gsutil only)."""

import json
import os
import random
import subprocess
import tempfile


def _sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def build_queue(bucket, jobs):
    """Write the full job manifest to GCS. Overwrites any prior manifest.
    `jobs` is the full list; workers skip ones already done via results/."""
    specs = [{"id": j["id"], "kind": j["kind"], "params": j["params"]}
             for j in jobs]
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(specs, tmp)
    tmp.close()
    _sh(["gsutil", "-q", "cp", tmp.name, f"gs://{bucket}/queue/manifest.json"])
    os.unlink(tmp.name)
    return len(specs)


def read_manifest(bucket):
    """Read the job manifest (static for a run)."""
    r = _sh(["gsutil", "cat", f"gs://{bucket}/queue/manifest.json"])
    if r.returncode != 0:
        return []
    return json.loads(r.stdout)


def done_ids(bucket):
    """Set of job ids whose result.json exists."""
    r = _sh(["gsutil", "ls", f"gs://{bucket}/results/*/result.json"])
    ids = set()
    for ln in r.stdout.splitlines():
        parts = ln.strip().split("/")
        if len(parts) >= 3 and parts[-1] == "result.json":
            ids.add(parts[-2])
    return ids


def claimed_ids(bucket):
    """Set of job ids that currently have a claim marker."""
    r = _sh(["gsutil", "ls", f"gs://{bucket}/queue/claims/"])
    ids = set()
    for ln in r.stdout.splitlines():
        ln = ln.strip()
        if ln.endswith(".json"):
            ids.add(os.path.basename(ln)[:-5])
    return ids


def try_claim(bucket, job_id, worker_id, vm):
    """Atomically claim `job_id`. Returns True iff this caller won the claim.
    Race-free via the if-generation-match:0 create precondition."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"worker": worker_id, "vm": vm, "job": job_id}, tmp)
    tmp.close()
    r = _sh(["gsutil", "-q", "-h", "x-goog-if-generation-match:0", "cp",
             tmp.name, f"gs://{bucket}/queue/claims/{job_id}.json"])
    os.unlink(tmp.name)
    return r.returncode == 0


def claim_next(bucket, worker_id, vm, manifest):
    """Claim and return the next available job spec, or None if none is
    currently claimable (either all done or all in-flight). Candidates are
    shuffled so concurrent workers rarely contend on the same job first."""
    done = done_ids(bucket)
    claimed = claimed_ids(bucket)
    cands = [j for j in manifest
             if j["id"] not in done and j["id"] not in claimed]
    random.shuffle(cands)
    for j in cands:
        if try_claim(bucket, j["id"], worker_id, vm):
            return j
    return None


def reap_stale_claims(bucket, alive_vms):
    """Delete claim markers whose owning VM is gone and whose job has no
    result.json — returning those jobs to the available pool. Returns the
    list of reaped job ids."""
    done = done_ids(bucket)
    r = _sh(["gsutil", "ls", f"gs://{bucket}/queue/claims/"])
    reaped = []
    for ln in r.stdout.splitlines():
        ln = ln.strip()
        if not ln.endswith(".json"):
            continue
        jid = os.path.basename(ln)[:-5]
        if jid in done:
            continue
        cr = _sh(["gsutil", "cat", ln])
        vm = None
        try:
            vm = json.loads(cr.stdout).get("vm")
        except Exception:
            pass
        if vm and vm not in alive_vms:
            _sh(["gsutil", "-q", "rm", ln])
            reaped.append(jid)
    return reaped
