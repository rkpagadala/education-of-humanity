# =============================================================================
# scripts/ml/cloud/worker_loop.py
#
# In-VM worker for the Chapter 9 worker pool. Boots once (driver + deps paid
# a single time by startup_worker.sh), then loops:
#
#     claim next job  ->  run it (isolated subprocess, with timeout)
#                     ->  upload outputs (result.json last = done marker)
#                     ->  repeat
#
# until the queue is drained, at which point it returns and the startup
# wrapper self-deletes the VM. Each job runs as a separate `run_job.py`
# subprocess so a crash, hang, or memory leak is isolated to one job.
#
# Environment (set by startup_worker.sh):
#   BUCKET, WORKER_ID, VM_NAME
# =============================================================================
"""worker_loop.py — persistent worker: claim/run/upload loop over the queue."""

import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import queue_lib as q

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

BUCKET = os.environ["BUCKET"]
WORKER_ID = os.environ.get("WORKER_ID", "w?")
VM_NAME = os.environ.get("VM_NAME", "?")

# Per-job wall-clock ceiling. Generous: the longest sharded job is ~1.75h.
# A job exceeding this is almost certainly hung; we record a timeout and the
# worker moves on rather than wedging forever.
JOB_TIMEOUT_SEC = int(os.environ.get("JOB_TIMEOUT_SEC", str(3 * 3600)))

# How long to keep polling when nothing is claimable before concluding the
# run is finished (other workers are draining the tail).
IDLE_POLL_SEC = 30
MAX_IDLE_POLLS = 20


def _upload(job_id, out_dir):
    """Upload a job's outputs. result.json is copied LAST so the done marker
    never appears before the artifacts it summarizes."""
    has_result = os.path.exists(os.path.join(out_dir, "result.json"))
    # Everything except result.json first.
    subprocess.run(
        f'gsutil -q -m rsync -r -x "result\\.json$" '
        f'"{out_dir}" "gs://{BUCKET}/results/{job_id}/"',
        shell=True,
    )
    if has_result:
        subprocess.run(
            ["gsutil", "-q", "cp", os.path.join(out_dir, "result.json"),
             f"gs://{BUCKET}/results/{job_id}/result.json"],
        )


def main():
    manifest = q.read_manifest(BUCKET)
    n_total = len(manifest)
    print(f"[worker {WORKER_ID}] manifest has {n_total} jobs", flush=True)

    ran = 0
    idle = 0
    while True:
        # Re-read the manifest each iteration so the queue can be extended or
        # re-sharded live without recycling running workers. Cheap relative to
        # the per-iteration done/claimed listing already done in claim_next.
        manifest = q.read_manifest(BUCKET) or manifest
        n_total = len(manifest)
        job = q.claim_next(BUCKET, WORKER_ID, VM_NAME, manifest)
        if job is None:
            done = q.done_ids(BUCKET)
            if len(done) >= n_total:
                print(f"[worker {WORKER_ID}] queue drained "
                      f"({len(done)}/{n_total} done); exiting", flush=True)
                break
            idle += 1
            if idle >= MAX_IDLE_POLLS:
                print(f"[worker {WORKER_ID}] idle {idle} polls with "
                      f"{len(done)}/{n_total} done; exiting", flush=True)
                break
            time.sleep(IDLE_POLL_SEC)
            continue

        idle = 0
        jid, kind, params = job["id"], job["kind"], job["params"]
        out_dir = os.path.join("/opt/job/output", jid)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[worker {WORKER_ID}] running {jid} (kind={kind})", flush=True)
        t0 = time.time()
        cmd = [
            "python3", os.path.join(SCRIPT_DIR, "run_job.py"),
            "--job-id", jid, "--kind", kind,
            "--params", json.dumps(params),
            "--output-dir", out_dir,
        ]
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, timeout=JOB_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            with open(os.path.join(out_dir, "error.json"), "w") as f:
                json.dump({"error": "timeout",
                           "timeout_sec": JOB_TIMEOUT_SEC}, f)
            print(f"[worker {WORKER_ID}] {jid} TIMED OUT", flush=True)
        _upload(jid, out_dir)
        ran += 1
        print(f"[worker {WORKER_ID}] finished {jid} in "
              f"{time.time() - t0:.0f}s (this worker ran {ran})", flush=True)
        # Free disk: drop the local copy now that it's uploaded.
        subprocess.run(["rm", "-rf", out_dir])

    print(f"[worker {WORKER_ID}] done; ran {ran} jobs", flush=True)


if __name__ == "__main__":
    main()
