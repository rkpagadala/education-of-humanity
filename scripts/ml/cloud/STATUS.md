# Panel ML — big run status

## What's running

**282 jobs** on spot T4 VMs in `us-central1-a` (project `$GCP_PROJECT`).

| Component | Jobs | Description |
|---|---:|---|
| Transformer T+25, 50 seeds | 50 | Joint multi-target headline |
| Transformer parent-vantage, 30 seeds × 3 outcomes | 90 | LE@12, TFR@5, U5MR@12 |
| Walk-forward, 8 cutoffs × 5 seeds | 40 | Train pre-cutoff, predict post |
| Region stratification, 6 regions × 5 seeds | 30 | Universality across regions |
| Era stratification, 5 eras × 5 seeds | 25 | Universality across decades |
| Income stratification, 3 tiers × 5 seeds | 15 | Universality across income |
| LOO-185 transformer | 1 | Each of 185 countries held out |
| Long-run 1875→2015 forecast | 10 | Century-scale forward prediction |
| Barro-Lee replication | 15 | Same finding on a different ed series |
| Spec curve (5 methods, CPU) | 1 | OLS, Ridge, Lasso, RF, GBM |
| Placebos (6 placebos × 4 methods, CPU) | 1 | Six falsification nulls |
| Double ML (CPU) | 1 | Chernozhukov cross-fit |
| Optuna search (500 trials, GPU) | 1 | Architecture not cherry-picked |
| Counterfactual swaps (CPU) | 1 | Korea/Philippines etc., from pretrained ckpts |
| Conditional permutation (CPU) | 1 | Alternative attribution |
| **Total** | **282** | |

Estimated: **~50 GPU-hours + 18 CPU-hours** (revised down after T4 turned out 12× faster than CPU).
At spot pricing: **~$5-15 expected, $30 ceiling with re-runs.**

Wall clock: 12-25 hours on 4 concurrent T4 quota.

## How to check progress

```bash
# Running VMs
gcloud compute instances list --filter="name~^ch9-" \
    --project="$GCP_PROJECT" \
    --format="value(name,status,scheduling.preemptible)"

# Completed jobs in GCS
gsutil ls "gs://$GCS_BUCKET/results/" | wc -l

# Tail the launcher log (if still attached)
tail -f /tmp/big_run_launch.log
```

## How to resume if launcher dies

The orchestration is idempotent — completed jobs are detected via
`result.json` in GCS. Re-running the launcher picks up where it left off:

```bash
cd <repo-root>
python3 scripts/ml/cloud/launch.py panel --skip-stage --max-concurrent 4
```

## When done

```bash
# Pull all results local
python3 scripts/ml/cloud/launch.py aggregate

# Build the final spec-curve table + headline numbers
python3 scripts/ml/chapter9/aggregate_results.py

# All cited numbers land in scripts/ml/checkin/chapter9_*.json
# (registered with verify_the_long_childhood.py in a follow-up step)
```

## Falsification checks built in

If any of these fire, the headline is wrong:
- Spec-curve agreement: all 5 methods must give 22-31% education R² drop
- All 6 placebos must return ~0 education R² drop
- Walk-forward at every cutoff must show > 0.7 R² on the held-out era
- Region stratification: no region's R² should be near zero
- LOO-185: removing any single country must not change the headline
- Counterfactual swaps for Korea/Philippines etc must reproduce the
  empirical outcome gap

## Cost monitoring

Per-hour spot pricing:
- T4 GPU + n1-standard-4: ~$0.105/hr
- n2-standard-8 CPU: ~$0.078/hr

If the bill goes above $40, something is wrong (likely preempt loop).
